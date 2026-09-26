use std::collections::{HashMap, VecDeque};

/// Maximum number of cached buffers per exact byte-size bucket.
///
/// 64 gives enough depth to absorb a full decode loop's worth of temporaries
/// (typically 8–32 per forward pass) while keeping the worst-case cached
/// footprint to 64× the largest recurring buffer — acceptable for VRAM.
pub(super) const FREE_LIST_CAP: usize = 64;

/// Default global cap on the TOTAL bytes retained by the free list, across all
/// size buckets.
///
/// The per-bucket [`FREE_LIST_CAP`] bounds a single size, but the number of
/// distinct sizes is unbounded: variable-length / many-shape workloads (e.g.
/// packed varlen embedding ingest) free buffers of continuously varying sizes,
/// each spawning a new bucket. Without a global cap the free list retains
/// `Σ (per-size buffers × size)` device memory indefinitely — the driver sees
/// those cached pointers as live, so free VRAM falls monotonically until OOM.
///
/// This cap bounds the total cached footprint; when exceeded, the oldest cached
/// buffers are returned to the driver. Overridable via the
/// `NUMR_CUDA_FREE_LIST_CAP_MB` environment variable (value in MiB).
const DEFAULT_FREE_LIST_CAP_BYTES: usize = 1024 * 1024 * 1024;

/// Resolve the global free-list byte cap from the environment (MiB) or default.
pub(super) fn resolve_free_list_cap_bytes() -> usize {
    super::super::env_config::env_mib_to_bytes(
        "NUMR_CUDA_FREE_LIST_CAP_MB",
        DEFAULT_FREE_LIST_CAP_BYTES as u64,
    ) as usize
}

/// Per-size free buffers plus a running total of cached bytes.
#[derive(Default)]
pub(super) struct FreeList {
    /// size_bytes → cached device pointers of that exact size (FIFO per bucket).
    pub(super) map: HashMap<u64, VecDeque<u64>>,
    /// Sum of `size × count` across every bucket — the live cached footprint.
    pub(super) total_bytes: usize,
}

impl FreeList {
    /// Pop a cached buffer of exactly `size_bytes`, updating the byte total.
    pub(super) fn pop(&mut self, size_bytes: u64) -> Option<u64> {
        let ptr = self.map.get_mut(&size_bytes).and_then(|b| b.pop_front())?;
        self.total_bytes -= size_bytes as usize;
        Some(ptr)
    }

    /// Push a cached buffer of `size_bytes`, updating the byte total.
    pub(super) fn push(&mut self, size_bytes: u64, ptr: u64) {
        self.map.entry(size_bytes).or_default().push_back(ptr);
        self.total_bytes += size_bytes as usize;
    }

    /// Evict cached buffers (largest size first) until `total_bytes <= cap`.
    /// Returns the evicted pointers for the caller to free outside the lock.
    pub(super) fn evict_to_cap(&mut self, cap: usize) -> Vec<u64> {
        let mut evicted = Vec::new();
        while self.total_bytes > cap {
            let largest = self
                .map
                .iter()
                .filter(|(_, b)| !b.is_empty())
                .map(|(&s, _)| s)
                .max();
            match largest {
                Some(size) => {
                    if let Some(ptr) = self.pop(size) {
                        evicted.push(ptr);
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }
        evicted
    }
}
