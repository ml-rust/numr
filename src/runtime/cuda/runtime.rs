//! CUDA runtime implementation

use super::cache::{
    get_or_create_client, is_cuda_context_valid, log_cuda_memory_error, reset_client,
    try_get_cached_stream,
};
use super::client::CudaAllocator;
use super::client::CudaClient;
use super::device::CudaDevice;
use super::kernels;
use crate::runtime::Runtime;

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
    type RawHandle = super::CudaRawHandle;

    fn name() -> &'static str {
        "cuda"
    }

    fn supports_graph_capture() -> bool {
        true // CUDA supports graph capture
    }

    /// Allocate GPU memory.
    ///
    /// Returns `Err(OutOfMemory)` if CUDA memory allocation fails.
    fn allocate(size_bytes: usize, device: &Self::Device) -> crate::error::Result<u64> {
        if size_bytes == 0 {
            return Ok(0);
        }

        let client = get_or_create_client(device);

        unsafe {
            let mut ptr: u64 = 0;
            let result = cudarc::driver::sys::cuMemAllocAsync(
                &mut ptr,
                size_bytes,
                client.stream.cu_stream(),
            );

            if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Ok(ptr);
            }

            // First attempt failed - try syncing the stream to flush pending frees
            let _ = client.stream.synchronize();

            let result = cudarc::driver::sys::cuMemAllocAsync(
                &mut ptr,
                size_bytes,
                client.stream.cu_stream(),
            );

            if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Ok(ptr);
            }

            // Stream is likely in a sticky error state (e.g., CUDA_ERROR_MISALIGNED_ADDRESS
            // from a previous kernel). Reset the client with a fresh context/stream.
            drop(client);
            if let Some(new_client) = reset_client(device) {
                let result = cudarc::driver::sys::cuMemAllocAsync(
                    &mut ptr,
                    size_bytes,
                    new_client.stream.cu_stream(),
                );

                if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Ok(ptr);
                }
            }

            Err(crate::error::Error::OutOfMemory { size: size_bytes })
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
    /// Returns an error if the CUDA copy operation fails.
    fn copy_to_device(src: &[u8], dst: u64, device: &Self::Device) -> crate::error::Result<()> {
        if src.is_empty() || dst == 0 {
            return Ok(());
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
                return Err(crate::error::Error::Backend(format!(
                    "CUDA host-to-device copy failed: {} bytes ({:?})",
                    src.len(),
                    result
                )));
            }

            // Synchronize to ensure data is available
            let _ = client.stream.synchronize();
        }
        Ok(())
    }

    /// Copy data from device to host.
    ///
    /// Returns an error if the CUDA copy operation fails.
    fn copy_from_device(
        src: u64,
        dst: &mut [u8],
        device: &Self::Device,
    ) -> crate::error::Result<()> {
        if dst.is_empty() || src == 0 {
            return Ok(());
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
                return Err(crate::error::Error::Backend(format!(
                    "[numr::cuda] Device-to-host copy failed: {} bytes ({:?})",
                    dst.len(),
                    result
                )));
            }

            // Synchronize to ensure data is available on host
            let _ = client.stream.synchronize();
        }
        Ok(())
    }

    /// Copy data within device memory.
    ///
    /// Returns an error if the CUDA copy operation fails.
    fn copy_within_device(
        src: u64,
        dst: u64,
        size_bytes: usize,
        device: &Self::Device,
    ) -> crate::error::Result<()> {
        if size_bytes == 0 || src == 0 || dst == 0 {
            return Ok(());
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
                return Err(crate::error::Error::Backend(format!(
                    "[numr::cuda] Device-to-device copy failed: {} bytes ({:?})",
                    size_bytes, result
                )));
            }
        }
        Ok(())
    }

    fn copy_strided(
        src_handle: u64,
        src_byte_offset: usize,
        dst_handle: u64,
        shape: &[usize],
        strides: &[isize],
        elem_size: usize,
        device: &Self::Device,
    ) -> crate::error::Result<()> {
        if src_handle == 0 || dst_handle == 0 || shape.is_empty() {
            return Ok(());
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(());
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
                return Err(crate::error::Error::Backend(format!(
                    "[numr::cuda] Failed to allocate shape array for strided copy ({:?})",
                    result
                )));
            }

            // Allocate device memory for strides array
            let mut strides_ptr: u64 = 0;
            let result =
                cudarc::driver::sys::cuMemAllocAsync(&mut strides_ptr, strides_bytes, cu_stream);
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                // Free shape_ptr before returning error
                let _ = cudarc::driver::sys::cuMemFreeAsync(shape_ptr, cu_stream);
                return Err(crate::error::Error::Backend(format!(
                    "[numr::cuda] Failed to allocate strides array for strided copy ({:?})",
                    result
                )));
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
                return Err(crate::error::Error::Backend(format!(
                    "[numr::cuda] Failed to copy shape to device for strided copy ({:?})",
                    result
                )));
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
                return Err(crate::error::Error::Backend(format!(
                    "[numr::cuda] Failed to copy strides to device for strided copy ({:?})",
                    result
                )));
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
                return Err(crate::error::Error::Backend(format!(
                    "[numr::cuda] Strided copy kernel failed: {} bytes ({} elements × {} bytes/elem) from {} to {} on device {}: {:?}",
                    numel * elem_size,
                    numel,
                    elem_size,
                    src_handle,
                    dst_handle,
                    device.index,
                    e
                )));
            }

            // Synchronize to ensure copy is complete
            let _ = client.stream.synchronize();
        }
        Ok(())
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

/// Get the default CUDA device (device 0)
pub fn cuda_device() -> CudaDevice {
    CudaDevice::new(0)
}

/// Get a specific CUDA device by ID
pub fn cuda_device_id(device_id: usize) -> CudaDevice {
    CudaDevice::new(device_id)
}

/// Check if CUDA is available on this system
pub fn is_cuda_available() -> bool {
    std::panic::catch_unwind(|| {
        let device = CudaDevice::new(0);
        let _client = get_or_create_client(&device);
    })
    .is_ok()
}
