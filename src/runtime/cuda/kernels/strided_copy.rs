//! Strided copy CUDA kernel launcher
//!
//! Provides GPU-accelerated strided-to-contiguous tensor copy operations.
//! This replaces the inefficient per-element cuMemcpy approach with a
//! parallel CUDA kernel.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, launch_config,
};
use crate::error::{Error, Result};

/// Module name for strided copy operations
pub const STRIDED_COPY_MODULE: &str = "strided_copy";

/// Maximum number of dimensions supported by the kernel
pub const MAX_DIMS: usize = 8;

/// Launch the strided copy kernel.
///
/// Copies non-contiguous (strided) tensor data to a contiguous destination buffer
/// using parallel GPU threads. Each thread handles one element.
///
/// # Safety
///
/// - `src_ptr` must be valid device memory
/// - `dst_ptr` must be valid device memory with space for `numel * elem_size` bytes
/// - `shape_ptr` must point to device memory containing `ndim` u64 values
/// - `strides_ptr` must point to device memory containing `ndim` i64 values
/// - All device memory must be allocated on the same device as the stream
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `src_ptr` - Source buffer device pointer
/// * `dst_ptr` - Destination buffer device pointer (contiguous)
/// * `shape_ptr` - Device pointer to shape array (u64[ndim])
/// * `strides_ptr` - Device pointer to strides array (i64[ndim])
/// * `numel` - Total number of elements
/// * `ndim` - Number of dimensions
/// * `elem_size` - Size of each element in bytes
/// * `src_byte_offset` - Byte offset into source buffer
pub unsafe fn launch_strided_copy(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    src_ptr: u64,
    dst_ptr: u64,
    shape_ptr: u64,
    strides_ptr: u64,
    numel: usize,
    ndim: usize,
    elem_size: usize,
    src_byte_offset: usize,
) -> Result<()> {
    if numel == 0 || ndim == 0 {
        return Ok(());
    }

    if ndim > MAX_DIMS {
        return Err(Error::Internal(format!(
            "strided_copy supports at most {} dimensions, got {}",
            MAX_DIMS, ndim
        )));
    }

    unsafe {
        let module = get_or_load_module(context, device_index, STRIDED_COPY_MODULE)?;
        let func = get_kernel_function(&module, "strided_copy")?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let numel_u32 = numel as u32;
        let ndim_u32 = ndim as u32;
        let elem_size_u32 = elem_size as u32;
        let src_offset_u64 = src_byte_offset as u64;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&dst_ptr);
        builder.arg(&shape_ptr);
        builder.arg(&strides_ptr);
        builder.arg(&numel_u32);
        builder.arg(&ndim_u32);
        builder.arg(&elem_size_u32);
        builder.arg(&src_offset_u64);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA strided_copy kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch the optimized 2D strided copy kernel.
///
/// For tensors with a simple 2D strided layout (outer dimension with stride,
/// inner dimension contiguous), this kernel is more efficient than the general
/// N-dimensional version.
///
/// # Safety
///
/// Same requirements as [`launch_strided_copy`].
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `src_ptr` - Source buffer device pointer
/// * `dst_ptr` - Destination buffer device pointer (contiguous)
/// * `outer_size` - Size of outer dimension
/// * `inner_size` - Size of inner (contiguous) dimension
/// * `outer_stride` - Stride of outer dimension (in elements)
/// * `elem_size` - Size of each element in bytes
/// * `src_byte_offset` - Byte offset into source buffer
#[allow(dead_code)] // Available for future optimization
pub unsafe fn launch_strided_copy_2d(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    src_ptr: u64,
    dst_ptr: u64,
    outer_size: usize,
    inner_size: usize,
    outer_stride: isize,
    elem_size: usize,
    src_byte_offset: usize,
) -> Result<()> {
    let numel = outer_size * inner_size;
    if numel == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, STRIDED_COPY_MODULE)?;
        let func = get_kernel_function(&module, "strided_copy_2d")?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let outer_size_u64 = outer_size as u64;
        let inner_size_u64 = inner_size as u64;
        let outer_stride_i64 = outer_stride as i64;
        let elem_size_u32 = elem_size as u32;
        let src_offset_u64 = src_byte_offset as u64;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&dst_ptr);
        builder.arg(&outer_size_u64);
        builder.arg(&inner_size_u64);
        builder.arg(&outer_stride_i64);
        builder.arg(&elem_size_u32);
        builder.arg(&src_offset_u64);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA strided_copy_2d kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}
