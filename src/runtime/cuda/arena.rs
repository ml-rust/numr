//! Bump-pointer arena for CUDA graph-internal intermediate allocations.

/// Bump-pointer arena for graph-internal intermediate allocations.
///
/// During CUDA graph capture, any allocation made via `cuMemAllocAsync` inside
/// the capture region produces a device address that is baked into the graph's
/// kernel-parameter block.  On replay those same addresses must be valid — but
/// `AUTO_FREE_ON_LAUNCH` frees graph-managed memory **after each launch**, so the
/// second replay finds freed addresses and faults with `CUDA_ERROR_ILLEGAL_ADDRESS`.
///
/// The arena solves this by redirecting every allocation made during the freeze
/// window into a single pre-allocated device buffer.  Because the buffer was
/// allocated **outside** the capture region (before `freeze()` was called), its
/// device address is stable: it persists across all graph replays for the
/// lifetime of the `CapturedGraph` that owns the arena tensor.
///
/// ## Allocation strategy
///
/// A bump pointer (`high_water`) advances **strictly monotonically**.  Each
/// allocation is aligned to 256 bytes (CUDA's minimum safe alignment for kernel
/// arguments).  Deallocation is bookkeeping-only: `high_water` is **never**
/// rewound, even for a top-of-stack free, because rewinding would let a later
/// allocation hand out a device address already baked into a captured kernel
/// node (see [`CudaArena::deallocate`] for the full rationale).  The arena is
/// bounded by `size` bytes; if capacity is exceeded `allocate()` returns
/// `Err(OutOfMemory)`.
///
/// ## Lifetime
///
/// The arena struct lives inside `Arc<Mutex<Option<CudaArena>>>` on the
/// allocator.  It is installed before `freeze()` via `install_arena()` and
/// cleared inside `unfreeze()` via `clear_arena()`.  The device buffer itself
/// is owned by the `Tensor<CudaRuntime>` held in `CapturedGraph::arena` — it
/// is freed only when `CapturedGraph` is dropped, which is always AFTER the
/// `CudaGraph` handle is destroyed (declaration-order drop).
pub(super) struct CudaArena {
    /// Base device address of the pre-allocated arena buffer.
    base: u64,
    /// Total capacity of the arena in bytes.
    size: usize,
    /// Offset of the next free byte (bump pointer).  Always ≤ `size`.
    high_water: usize,
    /// (offset, aligned_size) of every live region, in allocation order.
    ///
    /// Maintained purely for bookkeeping (debug diagnostics, `live_regions`
    /// length assertions). It does NOT drive capacity reclaim — `high_water`
    /// is strictly monotone regardless of what is removed here.
    live_regions: Vec<(usize, usize)>,
}

impl CudaArena {
    pub(super) fn new(base: u64, size: usize) -> Self {
        Self {
            base,
            size,
            high_water: 0,
            live_regions: Vec::new(),
        }
    }

    /// Align `n` up to a multiple of 256 bytes (CUDA memory alignment).
    fn align_up(n: usize) -> usize {
        const ALIGN: usize = 256;
        (n + ALIGN - 1) & !(ALIGN - 1)
    }

    /// Bump-allocate `size_bytes` from the arena.
    ///
    /// Returns `Ok(device_ptr)` on success or `Err(OutOfMemory)` if the arena
    /// would overflow.
    pub(super) fn allocate(&mut self, size_bytes: usize) -> crate::error::Result<u64> {
        let aligned = Self::align_up(size_bytes.max(1));
        if self.high_water + aligned > self.size {
            return Err(crate::error::Error::OutOfMemory { size: size_bytes });
        }
        let offset = self.high_water;
        self.high_water += aligned;
        self.live_regions.push((offset, aligned));
        Ok(self.base + offset as u64)
    }

    /// Mark an arena region as logically freed during a graph capture session.
    ///
    /// # Why we NEVER rewind `high_water`
    ///
    /// During CUDA graph capture, every `allocate()` call bakes an arena address
    /// into the graph's kernel-parameter blocks.  If we allowed LIFO reclaim
    /// (rewinding `high_water` on a top-of-stack free), a subsequent `allocate()`
    /// call could hand out the **same device address** to a different logical
    /// tensor.  The graph would then contain two distinct kernel nodes that both
    /// reference the same physical address — one writing its output there while
    /// the other reads from it under the assumption it holds its own data.  This
    /// silent aliasing causes `CUDA_ERROR_ILLEGAL_ADDRESS` (or worse, silent
    /// data corruption) on graph replay.
    ///
    /// The correct policy for a CUDA-graph arena is **strictly monotone bump
    /// allocation**: addresses only ever increase; freed regions stay "reserved"
    /// until the entire capture session ends and the arena is reset by
    /// `clear_arena()`.  Memory fragmentation is acceptable here because graph
    /// capture sessions are short and the arena is sized for worst-case peak live
    /// working set, not total allocation volume.
    ///
    /// `live_regions` is still maintained for bookkeeping (debug assertions,
    /// future diagnostics) but `high_water` is intentionally never decreased.
    pub(super) fn deallocate(&mut self, ptr: u64) {
        let offset = match ptr.checked_sub(self.base) {
            Some(o) => o as usize,
            None => return, // ptr is not inside this arena — ignore
        };

        // Find and remove the tracking entry.  This is bookkeeping only —
        // high_water is NOT rewound.  See the method-level comment for why.
        if let Some(idx) = self
            .live_regions
            .iter()
            .rposition(|(off, _)| *off == offset)
        {
            self.live_regions.remove(idx);
        }
        // Intentionally no rewind of high_water: addresses must be unique
        // across the entire capture session so graph kernel nodes never alias.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the arena bump-pointer logic using synthetic (host-side) addresses.
    ///
    /// This test constructs a `CudaArena` directly with a fictitious base address
    /// so it runs on any machine without a GPU.  It validates:
    ///
    /// - Consecutive allocations return contiguous, 256-aligned offsets from `base`.
    /// - Deallocate removes entries from `live_regions` (bookkeeping) but does
    ///   NOT rewind `high_water` — the arena is strictly monotone to prevent
    ///   device-address aliasing inside CUDA graph capture regions.
    /// - OOM is returned when the arena is full.
    #[test]
    fn arena_bump_pointer_logic_no_gpu() {
        // Use a synthetic base address — no real GPU memory needed.
        let base: u64 = 0x1_0000_0000;
        let arena_size: usize = 4096;
        let mut arena = CudaArena::new(base, arena_size);

        // First allocation: 100 bytes → aligned to 256 bytes.
        let p0 = arena.allocate(100).expect("alloc 100 bytes");
        assert_eq!(p0, base, "first alloc must start at base");
        assert_eq!(arena.high_water, 256);
        assert_eq!(arena.live_regions.len(), 1);

        // Second allocation: 512 bytes → aligned to 512 bytes.
        let p1 = arena.allocate(512).expect("alloc 512 bytes");
        assert_eq!(
            p1,
            base + 256,
            "second alloc starts after first (256 bytes)"
        );
        assert_eq!(arena.high_water, 768);
        assert_eq!(arena.live_regions.len(), 2);

        // Deallocate p1 (the topmost allocation).
        // high_water must NOT rewind — the arena is monotone.
        arena.deallocate(p1);
        assert_eq!(
            arena.high_water, 768,
            "deallocate must NOT rewind high_water (monotone arena)"
        );
        assert_eq!(
            arena.live_regions.len(),
            1,
            "p1 tracking entry removed; p0 remains"
        );

        // Third allocation: after p1 was logically freed, a new alloc advances
        // high_water further (no reuse of p1's address range).
        let p2 = arena.allocate(256).expect("alloc p2");
        assert_eq!(p2, base + 768, "p2 starts at high_water after p1 was freed");
        assert_eq!(arena.high_water, 1024);

        // Deallocate p0 (non-topmost). high_water must not rewind.
        arena.deallocate(p0);
        assert_eq!(
            arena.high_water, 1024,
            "non-topmost dealloc must NOT rewind high_water"
        );
        assert_eq!(
            arena.live_regions.len(),
            1,
            "p0 tracking entry removed; p2 remains"
        );

        // Deallocate p2 (topmost). Still must not rewind.
        arena.deallocate(p2);
        assert_eq!(
            arena.high_water, 1024,
            "even topmost dealloc must NOT rewind high_water (monotone policy)"
        );
        assert_eq!(arena.live_regions.len(), 0, "all tracking entries removed");

        // Fill the rest of the arena to trigger OOM.
        let remaining = arena_size - arena.high_water;
        let _ = arena.allocate(remaining).expect("fill remaining space");
        let oom = arena.allocate(1);
        assert!(
            oom.is_err(),
            "allocation beyond arena capacity must return OOM"
        );
    }
}
