//! Global client cache for CUDA runtime

use super::client::CudaClient;
use super::device::CudaDevice;
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

/// Global client cache: device index -> cached CudaClient
///
/// This caches CudaClient instances per device to avoid creating new
/// CUDA contexts, streams, and cuBLAS handles on every operation.
static CLIENT_CACHE: OnceLock<Mutex<HashMap<usize, CudaClient>>> = OnceLock::new();

/// Check if the CUDA context on the current thread is valid.
///
/// # Safety
///
/// This function calls CUDA driver API directly. It is safe to call at any time
/// but the result is only valid for the current thread's context state.
#[inline]
pub(super) unsafe fn is_cuda_context_valid() -> bool {
    let mut ctx: cudarc::driver::sys::CUcontext = std::ptr::null_mut();
    // SAFETY: cuCtxGetCurrent is safe to call at any time and writes to the provided pointer.
    let result = unsafe { cudarc::driver::sys::cuCtxGetCurrent(&mut ctx) };
    result == cudarc::driver::sys::CUresult::CUDA_SUCCESS && !ctx.is_null()
}

/// Safely lock the client cache, recovering from poisoned mutex.
///
/// If the mutex is poisoned (a thread panicked while holding the lock),
/// we recover the guard since our cache operations are idempotent.
#[inline]
fn lock_client_cache(
    cache: &Mutex<HashMap<usize, CudaClient>>,
) -> MutexGuard<'_, HashMap<usize, CudaClient>> {
    cache.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Get or create a cached CudaClient for a device.
pub(super) fn get_or_create_client(device: &CudaDevice) -> CudaClient {
    let cache = CLIENT_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache_guard = lock_client_cache(cache);

    if let Some(client) = cache_guard.get(&device.index) {
        return client.clone();
    }

    // Create new client and cache it
    let client = CudaClient::new(device.clone()).expect("Failed to create CUDA client");
    cache_guard.insert(device.index, client.clone());

    client
}

/// Try to get the stream from a cached client for a device.
///
/// Returns `None` if no client is cached or if the cache lock is unavailable.
#[inline]
pub(super) fn try_get_cached_stream(device_index: usize) -> Option<cudarc::driver::sys::CUstream> {
    let cache = CLIENT_CACHE.get()?;
    let guard = lock_client_cache(cache);
    guard
        .get(&device_index)
        .map(|client| client.stream.cu_stream())
}

/// Log a CUDA memory operation failure.
///
/// Uses stderr with a consistent prefix for easy filtering.
#[cold]
#[inline(never)]
pub(super) fn log_cuda_memory_error(
    operation: &str,
    ptr: u64,
    result: cudarc::driver::sys::CUresult,
) {
    eprintln!(
        "[numr::cuda] {} failed for ptr 0x{:x}: {:?}",
        operation, ptr, result
    );
}
