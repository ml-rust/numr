//! CUDA runtime implementation
//!
//! This module provides GPU acceleration via NVIDIA CUDA using cudarc.
//!
//! # Features
//!
//! - `CudaDevice` - Represents a CUDA GPU device
//! - `CudaClient` - Manages GPU stream and context, launches kernels
//! - `CudaRuntime` - Implements the generic Runtime trait
//! - `TensorOps` - CUDA-accelerated tensor operations using cuBLAS
//!
//! # Panics
//!
//! The following operations may panic on CUDA errors (allocation failures are
//! typically unrecoverable in GPU contexts):
//!
//! - `Runtime::allocate` - Panics if CUDA memory allocation fails
//! - `Runtime::copy_to_device` - Panics if host-to-device copy fails
//! - `Runtime::copy_from_device` - Panics if device-to-host copy fails
//! - `Runtime::copy_within_device` - Panics if device-to-device copy fails
//!
//! These panics follow CUDA best practices where allocation failures indicate
//! an unrecoverable out-of-memory condition.

mod client;
mod device;
mod kernels;
mod linalg;
mod ops;
#[cfg(feature = "sparse")]
mod sparse;

#[cfg(feature = "sparse")]
pub use crate::sparse::SparseOps;
pub use client::{CudaAllocator, CudaClient, CudaRawHandle};
pub use device::{CudaDevice, CudaError};

use crate::runtime::Runtime;
use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

// ============================================================================
// Internal Helpers
// ============================================================================

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
unsafe fn is_cuda_context_valid() -> bool {
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
fn get_or_create_client(device: &CudaDevice) -> CudaClient {
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
fn try_get_cached_stream(device_index: usize) -> Option<cudarc::driver::sys::CUstream> {
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
fn log_cuda_memory_error(operation: &str, ptr: u64, result: cudarc::driver::sys::CUresult) {
    eprintln!(
        "[numr::cuda] {} failed for ptr 0x{:x}: {:?}",
        operation, ptr, result
    );
}

// ============================================================================
// Runtime Implementation
// ============================================================================

/// CUDA Runtime adapter
///
/// Implements the generic Runtime trait for CUDA backend.
/// Uses cudarc for direct GPU control.
#[derive(Clone, Debug, Default)]
pub struct CudaRuntime;

impl Runtime for CudaRuntime {
    type Device = CudaDevice;
    type Client = CudaClient;
    type Allocator = CudaAllocator;
    type RawHandle = CudaRawHandle;

    fn name() -> &'static str {
        "cuda"
    }

    fn supports_graph_capture() -> bool {
        true // CUDA supports graph capture
    }

    /// Allocate GPU memory.
    ///
    /// # Panics
    ///
    /// Panics if CUDA memory allocation fails. This follows CUDA best practices
    /// where allocation failures typically indicate an unrecoverable OOM condition.
    fn allocate(size_bytes: usize, device: &Self::Device) -> u64 {
        if size_bytes == 0 {
            return 0;
        }

        let client = get_or_create_client(device);

        unsafe {
            let mut ptr: u64 = 0;
            let result = cudarc::driver::sys::cuMemAllocAsync(
                &mut ptr,
                size_bytes,
                client.stream.cu_stream(),
            );

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                panic!(
                    "[numr::cuda] Allocation failed: {} bytes on device {} ({:?})",
                    size_bytes, device.index, result
                );
            }

            ptr
        }
    }

    fn deallocate(ptr: u64, _size_bytes: usize, device: &Self::Device) {
        if ptr == 0 {
            return;
        }

        unsafe {
            // Check if CUDA context is still valid before attempting free
            if !is_cuda_context_valid() {
                // Context is gone - memory will be reclaimed by driver on context destruction
                return;
            }

            // Try to use stream-ordered async free if client is available
            let result = if let Some(stream) = try_get_cached_stream(device.index) {
                cudarc::driver::sys::cuMemFreeAsync(ptr, stream)
            } else {
                // Fallback to synchronous free
                cudarc::driver::sys::cuMemFree_v2(ptr)
            };

            // Log failures but don't panic - deallocation errors are typically benign
            // (e.g., double-free attempts, already-freed memory)
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS
                && result != cudarc::driver::sys::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS
            {
                log_cuda_memory_error("cuMemFree", ptr, result);
            }
        }
    }

    /// Copy data from host to device.
    ///
    /// # Panics
    ///
    /// Panics if the CUDA copy operation fails.
    fn copy_to_device(src: &[u8], dst: u64, device: &Self::Device) {
        if src.is_empty() || dst == 0 {
            return;
        }

        let client = get_or_create_client(device);

        unsafe {
            let result = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                dst,
                src.as_ptr() as *const std::ffi::c_void,
                src.len(),
                client.stream.cu_stream(),
            );

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                panic!(
                    "[numr::cuda] Host-to-device copy failed: {} bytes ({:?})",
                    src.len(),
                    result
                );
            }

            // Synchronize to ensure data is available
            let _ = client.stream.synchronize();
        }
    }

    /// Copy data from device to host.
    ///
    /// # Panics
    ///
    /// Panics if the CUDA copy operation fails.
    fn copy_from_device(src: u64, dst: &mut [u8], device: &Self::Device) {
        if dst.is_empty() || src == 0 {
            return;
        }

        let client = get_or_create_client(device);

        unsafe {
            let result = cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                src,
                dst.len(),
                client.stream.cu_stream(),
            );

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                panic!(
                    "[numr::cuda] Device-to-host copy failed: {} bytes ({:?})",
                    dst.len(),
                    result
                );
            }

            // Synchronize to ensure data is available on host
            let _ = client.stream.synchronize();
        }
    }

    /// Copy data within device memory.
    ///
    /// # Panics
    ///
    /// Panics if the CUDA copy operation fails.
    fn copy_within_device(src: u64, dst: u64, size_bytes: usize, device: &Self::Device) {
        if size_bytes == 0 || src == 0 || dst == 0 {
            return;
        }

        let client = get_or_create_client(device);

        unsafe {
            let result = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                dst,
                src,
                size_bytes,
                client.stream.cu_stream(),
            );

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                panic!(
                    "[numr::cuda] Device-to-device copy failed: {} bytes ({:?})",
                    size_bytes, result
                );
            }
        }
    }

    fn copy_strided(
        src_handle: u64,
        src_byte_offset: usize,
        dst_handle: u64,
        shape: &[usize],
        strides: &[isize],
        elem_size: usize,
        device: &Self::Device,
    ) {
        if src_handle == 0 || dst_handle == 0 || shape.is_empty() {
            return;
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return;
        }

        let ndim = shape.len();
        let client = get_or_create_client(device);
        let cu_stream = client.stream.cu_stream();

        // Convert shape and strides to device-compatible types
        let shape_u64: Vec<u64> = shape.iter().map(|&s| s as u64).collect();
        let strides_i64: Vec<i64> = strides.iter().map(|&s| s as i64).collect();

        // Allocate temporary device memory for shape and strides arrays
        let shape_bytes = ndim * std::mem::size_of::<u64>();
        let strides_bytes = ndim * std::mem::size_of::<i64>();

        unsafe {
            // Allocate device memory for shape array
            let mut shape_ptr: u64 = 0;
            let result =
                cudarc::driver::sys::cuMemAllocAsync(&mut shape_ptr, shape_bytes, cu_stream);
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                panic!(
                    "[numr::cuda] Failed to allocate shape array for strided copy ({:?})",
                    result
                );
            }

            // Allocate device memory for strides array
            let mut strides_ptr: u64 = 0;
            let result =
                cudarc::driver::sys::cuMemAllocAsync(&mut strides_ptr, strides_bytes, cu_stream);
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                // Free shape_ptr before panicking
                let _ = cudarc::driver::sys::cuMemFreeAsync(shape_ptr, cu_stream);
                panic!(
                    "[numr::cuda] Failed to allocate strides array for strided copy ({:?})",
                    result
                );
            }

            // Copy shape to device
            let result = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                shape_ptr,
                shape_u64.as_ptr() as *const std::ffi::c_void,
                shape_bytes,
                cu_stream,
            );
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                let _ = cudarc::driver::sys::cuMemFreeAsync(shape_ptr, cu_stream);
                let _ = cudarc::driver::sys::cuMemFreeAsync(strides_ptr, cu_stream);
                panic!(
                    "[numr::cuda] Failed to copy shape to device for strided copy ({:?})",
                    result
                );
            }

            // Copy strides to device
            let result = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                strides_ptr,
                strides_i64.as_ptr() as *const std::ffi::c_void,
                strides_bytes,
                cu_stream,
            );
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                let _ = cudarc::driver::sys::cuMemFreeAsync(shape_ptr, cu_stream);
                let _ = cudarc::driver::sys::cuMemFreeAsync(strides_ptr, cu_stream);
                panic!(
                    "[numr::cuda] Failed to copy strides to device for strided copy ({:?})",
                    result
                );
            }

            // Launch the strided copy kernel
            let kernel_result = kernels::launch_strided_copy(
                &client.context,
                &client.stream,
                device.index,
                src_handle,
                dst_handle,
                shape_ptr,
                strides_ptr,
                numel,
                ndim,
                elem_size,
                src_byte_offset,
            );

            // Free temporary device memory (async, will happen after kernel completes)
            let _ = cudarc::driver::sys::cuMemFreeAsync(shape_ptr, cu_stream);
            let _ = cudarc::driver::sys::cuMemFreeAsync(strides_ptr, cu_stream);

            // Check kernel launch result
            if let Err(e) = kernel_result {
                panic!("[numr::cuda] Strided copy kernel failed: {:?}", e);
            }

            // Synchronize to ensure copy is complete
            let _ = client.stream.synchronize();
        }
    }

    fn default_device() -> Self::Device {
        CudaDevice::new(0)
    }

    fn default_client(device: &Self::Device) -> Self::Client {
        get_or_create_client(device)
    }

    fn raw_handle(client: &Self::Client) -> &Self::RawHandle {
        &client.raw_handle
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Get the default CUDA device (device 0)
pub fn cuda_device() -> CudaDevice {
    CudaDevice::new(0)
}

/// Get a specific CUDA device by ID
pub fn cuda_device_id(device_id: usize) -> CudaDevice {
    CudaDevice::new(device_id)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{Allocator, Device, RuntimeClient};

    #[test]
    fn test_cuda_device_creation() {
        let device = CudaDevice::new(0);
        assert_eq!(device.id(), 0);
        assert_eq!(device.name(), "cuda:0");
    }

    #[test]
    fn test_cuda_allocate_deallocate() {
        let device = CudaDevice::new(0);
        let ptr = CudaRuntime::allocate(1024, &device);
        assert_ne!(ptr, 0);
        CudaRuntime::deallocate(ptr, 1024, &device);
    }

    #[test]
    fn test_cuda_copy_roundtrip() {
        let device = CudaDevice::new(0);
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let ptr = CudaRuntime::allocate(data.len(), &device);
        CudaRuntime::copy_to_device(&data, ptr, &device);

        let mut result = vec![0u8; data.len()];
        CudaRuntime::copy_from_device(ptr, &mut result, &device);

        assert_eq!(data, result);

        CudaRuntime::deallocate(ptr, data.len(), &device);
    }

    #[test]
    fn test_cuda_copy_within_device() {
        let device = CudaDevice::new(0);
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let src = CudaRuntime::allocate(data.len(), &device);
        let dst = CudaRuntime::allocate(data.len(), &device);

        CudaRuntime::copy_to_device(&data, src, &device);
        CudaRuntime::copy_within_device(src, dst, data.len(), &device);

        let mut result = vec![0u8; data.len()];
        CudaRuntime::copy_from_device(dst, &mut result, &device);

        assert_eq!(data, result);

        CudaRuntime::deallocate(src, data.len(), &device);
        CudaRuntime::deallocate(dst, data.len(), &device);
    }

    #[test]
    fn test_cuda_client_creation() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);
        assert_eq!(client.device().id(), 0);
    }

    #[test]
    fn test_cuda_client_allocator() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let ptr = client.allocator().allocate(256);
        assert_ne!(ptr, 0);
        client.allocator().deallocate(ptr, 256);
    }

    #[test]
    fn test_cuda_compute_capability() {
        let device = CudaDevice::new(0);
        // Initialize CUDA context by getting the client
        let _client = CudaRuntime::default_client(&device);
        let (major, minor) = device
            .compute_capability()
            .expect("Failed to get compute capability");
        // Any modern GPU should be at least Maxwell (5.x) or newer
        assert!(
            major >= 5,
            "Expected compute capability >= 5.0, got {}.{}",
            major,
            minor
        );
    }

    #[test]
    fn test_cuda_memory_info() {
        let device = CudaDevice::new(0);
        // Initialize CUDA context by getting the client
        let client = CudaRuntime::default_client(&device);
        // Activate the context on this thread
        client
            .context
            .bind_to_thread()
            .expect("Failed to bind context");
        let (free, total) = device.memory_info().expect("Failed to get memory info");
        assert!(total > 0, "Total GPU memory should be > 0");
        assert!(free <= total, "Free memory should be <= total memory");
    }
}
