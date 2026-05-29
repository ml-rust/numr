//! Persistent device-side cache of Sobol direction vectors.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Persistent device-side cache of Sobol direction vectors.
///
/// Sobol direction vectors are computed once per unique `dimension` value and
/// uploaded to a device buffer that lives for the lifetime of the `CudaClient`.
/// Because the buffer is allocated **outside** any CUDA graph capture region,
/// its device address is stable across `cuGraphLaunch` replays — unlike a
/// per-call H2D copy, which embeds a `cuMemcpyHtoD` memcpy node whose source
/// host pointer may point to freed stack memory on replay.
///
/// ## Thread safety
///
/// All accesses are guarded by `Mutex<HashMap<...>>`. The struct is wrapped in
/// `Arc` so all `CudaClient` clones share the same underlying cache.
///
/// ## Lifetime and cleanup
///
/// The cache holds raw device pointers allocated via `cuMemAllocAsync` on the
/// client's compute stream. `Drop` drains the map and frees each buffer
/// synchronously via `cuMemFree_v2`. The synchronous free is intentional: by
/// the time the last `Arc<SobolDvCache>` is dropped, the client (and its
/// stream) may already have been torn down, so an async free is unsafe.
pub(crate) struct SobolDvCache {
    /// dimension → (device_ptr, num_u32s)
    inner: Mutex<HashMap<u32, (u64, usize)>>,
}

impl SobolDvCache {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(HashMap::new()),
        })
    }

    /// Look up a cached entry for `dimension`. Returns `(ptr, num_u32s)` if
    /// present.
    pub fn get(&self, dimension: u32) -> Option<(u64, usize)> {
        let map = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        map.get(&dimension).copied()
    }

    /// Insert a new cache entry. Does nothing if the dimension is already
    /// present (idempotent).
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid device pointer owning `num_u32s * 4` bytes
    /// allocated via `cuMemAllocAsync`. Ownership is transferred to the cache;
    /// the caller must not free the pointer.
    pub unsafe fn insert(&self, dimension: u32, ptr: u64, num_u32s: usize) {
        let mut map = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        // Only insert if not already present (first writer wins, idempotent).
        map.entry(dimension).or_insert((ptr, num_u32s));
    }
}

impl Drop for SobolDvCache {
    fn drop(&mut self) {
        // Drain the map and synchronously free every device buffer.
        //
        // We use `cuMemFree_v2` (synchronous) rather than `cuMemFreeAsync`
        // because at drop time the associated stream may already be destroyed.
        // `cuMemFree_v2` works on any pointer regardless of how it was
        // originally allocated (async pool or otherwise).
        let mut map = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        for (ptr, _size) in map.drain().map(|(_, v)| v) {
            if ptr != 0 {
                unsafe {
                    // At shutdown the CUDA context may already be torn down,
                    // which causes CUDA_ERROR_INVALID_CONTEXT — expected and
                    // benign. In debug builds we surface any *other* failure so
                    // a genuine leak (e.g. a driver bug) is not silent.
                    let r = cudarc::driver::sys::cuMemFree_v2(ptr);
                    #[cfg(debug_assertions)]
                    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS
                        && r != cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_CONTEXT
                        && r != cudarc::driver::sys::CUresult::CUDA_ERROR_DEINITIALIZED
                    {
                        eprintln!(
                            "[numr::cuda] SobolDvCache::drop: cuMemFree_v2 failed for \
                             0x{ptr:x}: {r:?} (device buffer leaked)"
                        );
                    }
                    #[cfg(not(debug_assertions))]
                    let _ = r;
                }
            }
        }
    }
}
