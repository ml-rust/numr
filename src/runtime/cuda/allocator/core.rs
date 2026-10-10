use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use super::super::arena::CudaArena;
use super::super::capture::{GuardedStream, thread_is_capturing};
use super::free_list::{FreeList, resolve_free_list_cap_bytes};

/// CUDA stream-ordered allocator with a Rust-side free list.
///
/// Combines a per-size-bucket Rust free list (fast path: no driver call) with
/// `cuMemAllocAsync`/`cuMemFreeAsync` for cold misses and pool management.
///
/// # Safety invariant — single canonical stream
///
/// Every pointer in the free list was allocated on `self.stream`. Every
/// deallocation is also issued on `self.stream`. Because all alloc/free ops
/// share the same stream, there is no cross-stream ordering hazard: a buffer
/// popped from the free list is guaranteed to have no pending use on any
/// other stream before it is returned to the caller.
///
/// This invariant holds because `CudaClient::new` always returns the cached
/// canonical client for a device (via `register_or_get_client`), and stream
/// creation only happens in `new_uncached`. The `copy_stream` is exclusively
/// for D2H copies and never used with the allocator.
///
/// # Pool Threshold
///
/// The default memory pool's release threshold is set to 512 MiB at context
/// creation (see `CudaClient` construction in `client.rs`): freed segments up
/// to that size stay warm for reuse in decode loops, while larger freed
/// segments are returned to the OS to avoid address-space fragmentation on
/// many-shape workloads. The threshold is overridable via the
/// `NUMR_CUDA_POOL_RELEASE_THRESHOLD_MB` environment variable (value in MiB).
/// The OOM-retry path additionally trims the pool to 0.
#[derive(Clone)]
pub struct CudaAllocator {
    /// The device's compute stream. `cuMemAllocAsync` and `cuMemFreeAsync`
    /// are stream-ordered enqueues, so the driver paths below take the
    /// device's capture lock through it.
    pub(super) stream: GuardedStream,
    /// Per-size free list with a running cached-byte total.
    ///
    /// VecDeque gives O(1) push_back / pop_front so the oldest entry is
    /// evicted first when the per-bucket cap is reached (FIFO within a bucket).
    /// A global byte cap ([`free_list_cap_bytes`](Self::free_list_cap_bytes))
    /// bounds the total across all buckets so many-shape workloads cannot retain
    /// device memory without bound.
    ///
    /// Every lock acquisition recovers from poisoning via `into_inner()`. The
    /// free list is a cache, not correctness-critical state: a panic mid-mutation
    /// can only desynchronise `total_bytes` from the bucket contents (over- or
    /// under-retention), never duplicate a pointer or publish an invalid one.
    /// Refusing to allocate for the rest of the process is strictly worse.
    pub(super) free_list: Arc<Mutex<FreeList>>,
    /// Global cap on total bytes retained by `free_list` across all buckets.
    pub(super) free_list_cap_bytes: usize,
    /// When frozen, alloc/free go directly to the driver to create proper
    /// CUDA graph alloc/free nodes — bypassing the Rust free list.
    pub(super) frozen: Arc<std::sync::atomic::AtomicBool>,
    /// Pointers allocated during the current freeze window.
    ///
    /// Every pointer returned by `allocate()` while `frozen` is true is inserted
    /// here. At `unfreeze()` time we verify that none of these addresses ended up
    /// inside `free_list` (which would mean a frozen-allocated pointer was
    /// accidentally routed through the un-frozen `deallocate()` path and absorbed
    /// into the Rust cache — a silent graph-state corruption).
    ///
    /// The set is cleared at `unfreeze()`. In debug builds the check panics;
    /// in release builds the assertion compiles away but the set is still
    /// maintained and cleared so the bookkeeping remains coherent.
    pub(super) captured_ptrs: Arc<Mutex<HashSet<u64>>>,
    /// Cached handle to the default memory pool for the device. Stored as
    /// `u64` (the raw pointer value) so the struct stays `Send + Sync`.
    /// Used only by the OOM retry path to call `cuMemPoolTrimTo`.
    /// Zero when the pool could not be obtained at construction.
    pool_handle: u64,
    /// Optional bump-pointer arena for graph-internal intermediates.
    ///
    /// When `Some`, the frozen `allocate()` path returns offsets into this
    /// pre-allocated device buffer instead of calling `cuMemAllocAsync`.
    /// This gives captured-kernel arguments stable device addresses that
    /// survive across `cuGraphLaunch` replays.
    ///
    /// Installed via `install_arena()` before `freeze()`.
    /// Cleared by `unfreeze()` via `clear_arena()`.
    pub(super) arena: Arc<Mutex<Option<CudaArena>>>,
}

impl CudaAllocator {
    /// Construct an allocator bound to `stream`.
    ///
    /// `pool_handle` is the raw `CUmemoryPool` pointer value for the device's
    /// default pool (zero if it could not be obtained), used only by the OOM
    /// retry path.
    pub(in crate::runtime::cuda) fn new(stream: GuardedStream, pool_handle: u64) -> Self {
        Self {
            stream,
            free_list: Arc::new(Mutex::new(FreeList::default())),
            free_list_cap_bytes: resolve_free_list_cap_bytes(),
            frozen: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            captured_ptrs: Arc::new(Mutex::new(HashSet::new())),
            pool_handle,
            arena: Arc::new(Mutex::new(None)),
        }
    }

    /// Whether THIS thread is the one capturing a graph on this device.
    ///
    /// The freeze flag alone is not the question. Every thread on the device
    /// shares it, while the graph arena and `captured_ptrs` belong to the ONE
    /// thread inside the capture region. Serving another thread from that
    /// arena hands two threads the same device address. Recording its pointer
    /// as graph-owned reads back as corruption, because the ordinary path
    /// frees it.
    ///
    /// Both conditions are required. The thread-local says who is inside a
    /// capture region. The flag says the allocator is serving one, which
    /// `freeze` and `unfreeze` bracket.
    pub(super) fn thread_owns_capture(&self) -> bool {
        self.frozen.load(std::sync::atomic::Ordering::Relaxed)
            && thread_is_capturing(self.stream.device_index())
    }

    /// Allocate directly from the driver (no free-list lookup).
    ///
    /// On failure, drains the Rust free list back to the driver pool so those
    /// segments become available for reuse, syncs the stream, then retries once.
    pub(super) unsafe fn driver_alloc(&self, size_bytes: usize) -> crate::error::Result<u64> {
        // One permit covers the alloc, the drain, the sync and the retry:
        // every one of them is stream-ordered on the compute stream.
        let _permit = self.stream.enqueue_permit()?;
        let cu_stream = self.stream.raw().cu_stream();
        let mut ptr: u64 = 0;
        let result =
            unsafe { cudarc::driver::sys::cuMemAllocAsync(&mut ptr, size_bytes, cu_stream) };
        if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Ok(ptr);
        }
        if result != cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
            return Err(alloc_error(size_bytes, result));
        }

        // Drain free list: return cached segments to the driver pool so it can
        // reclaim VRAM that our Rust-side cache was holding "live" from the
        // driver's perspective.
        let drained: Vec<u64> = {
            let mut fl = self.free_list.lock().unwrap_or_else(|p| p.into_inner());
            fl.total_bytes = 0;
            fl.map
                .drain()
                .flat_map(|(_, bucket)| bucket.into_iter())
                .collect()
        };
        for p in drained {
            let _ = unsafe { cudarc::driver::sys::cuMemFreeAsync(p, cu_stream) };
        }

        // Sync so the pool can process the frees before the retry alloc. The
        // permit taken at entry already covers this wait, so it goes through
        // the bare stream rather than taking a second read side.
        let _ = self.stream.raw().synchronize();

        // Trim the pool to release cached segments back to the OS. The
        // pool retains freed allocations (per the release threshold) which
        // is fast for tight decode loops but fragments the address space
        // when a one-shot large allocation arrives after many small frees.
        // Calling trim only on the retry path keeps the steady-state cache
        // behaviour but recovers from fragmentation when it matters.
        if self.pool_handle != 0 {
            let pool = self.pool_handle as cudarc::driver::sys::CUmemoryPool;
            let _ = unsafe { cudarc::driver::sys::cuMemPoolTrimTo(pool, 0) };
        }

        let result =
            unsafe { cudarc::driver::sys::cuMemAllocAsync(&mut ptr, size_bytes, cu_stream) };
        if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            Ok(ptr)
        } else {
            Err(alloc_error(size_bytes, result))
        }
    }

    /// Free directly to the driver (no free-list insertion).
    ///
    /// When the context cannot be made current the free cannot be issued, and
    /// the buffer stays allocated until the context is destroyed.
    pub(super) unsafe fn driver_free(&self, ptr: u64) {
        let _permit = match self.stream.enqueue_permit() {
            Ok(permit) => permit,
            Err(e) => {
                eprintln!(
                    "[numr::cuda] cuMemFreeAsync skipped for ptr 0x{ptr:x}: the CUDA context \
                     could not be made current on this thread ({e:?})"
                );
                return;
            }
        };
        let _ = unsafe { cudarc::driver::sys::cuMemFreeAsync(ptr, self.stream.raw().cu_stream()) };
    }

    /// Install a bump-pointer arena for the next freeze window.
    ///
    /// Must be called **before** [`crate::runtime::Allocator::freeze`]. `base` is the device
    /// address of a pre-allocated buffer of `size` bytes; both must remain
    /// valid until [`clear_arena`](Self::clear_arena) is called (which happens
    /// inside `unfreeze()`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`][crate::error::Error::Internal] if an arena is
    /// already installed. Graph capture on a single client is not re-entrant or
    /// thread-safe (the freeze flag and stream-capture state are shared), so a
    /// double install indicates two overlapping capture attempts — failing fast
    /// surfaces the misuse instead of silently overwriting the live arena.
    pub fn install_arena(&self, base: u64, size: usize) -> crate::error::Result<()> {
        let mut guard = self.arena.lock().unwrap_or_else(|p| p.into_inner());
        if guard.is_some() {
            return Err(crate::error::Error::Internal(
                "CudaAllocator::install_arena: an arena is already installed; \
                 graph capture is not re-entrant on a single client"
                    .into(),
            ));
        }
        *guard = Some(CudaArena::new(base, size));
        Ok(())
    }

    /// Remove the arena.
    ///
    /// Called automatically by `unfreeze()`. May also be called on error paths
    /// to discard an installed arena without going through a full freeze cycle.
    /// Dropping the bookkeeping does **not** free the device buffer — that is
    /// owned by the `Tensor<CudaRuntime>` held in `CapturedGraph`.
    pub fn clear_arena(&self) {
        let mut guard = self.arena.lock().unwrap_or_else(|p| p.into_inner());
        *guard = None;
    }

    /// Returns `true` if a bump-pointer arena is currently installed.
    pub fn has_arena(&self) -> bool {
        let guard = self.arena.lock().unwrap_or_else(|p| p.into_inner());
        guard.is_some()
    }

    /// Peak bytes the installed arena has handed out, or `None` when no
    /// arena is installed. Read this before `unfreeze()`, which clears the
    /// arena.
    pub fn arena_high_water(&self) -> Option<usize> {
        let guard = self.arena.lock().unwrap_or_else(|p| p.into_inner());
        guard.as_ref().map(CudaArena::high_water)
    }
}

/// The error for a failed `cuMemAllocAsync` of `size_bytes`.
///
/// Only the driver's out-of-memory code maps to `OutOfMemory`. Any other
/// code keeps its name, so a context or capture error is not reported as a
/// full device.
pub(in crate::runtime::cuda) fn alloc_error(
    size_bytes: usize,
    result: cudarc::driver::sys::CUresult,
) -> crate::error::Error {
    if result == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
        crate::error::Error::OutOfMemory { size: size_bytes }
    } else {
        crate::error::Error::Backend(format!(
            "[numr::cuda] cuMemAllocAsync of {size_bytes} bytes failed ({result:?})"
        ))
    }
}
