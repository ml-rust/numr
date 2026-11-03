//! Strided-to-strided copy CUDA kernel launcher
//!
//! Copies a strided source view into a strided destination view, one thread
//! per logical element. This backs `copy_into` with a non-contiguous `out`.
//!
//! Shape and both stride sets travel as kernel arguments (by value), the same
//! way [`launch_strided_copy`](super::strided_copy::launch_strided_copy) sends
//! them. The launch allocates nothing and copies nothing from the host, so it
//! records into a CUDA graph with stable addresses.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, launch_config,
};
use super::strided_copy::MAX_DIMS;
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;

/// Module name for strided-to-strided copy operations
pub const STRIDED_COPY_INTO_MODULE: &str = "strided_copy_into";

/// Launch the strided-to-strided copy kernel.
///
/// Element `i` of the logical index space moves from
/// `src + src_byte_offset + dot(i, src_strides) * elem_size` to
/// `dst + dst_byte_offset + dot(i, dst_strides) * elem_size`.
/// Strides are in elements and can be negative.
///
/// # Safety
///
/// - `src_ptr` and `dst_ptr` must be valid device memory on the stream's device.
/// - Every address the offsets and strides reach must lie inside its allocation.
/// - The element sets the two views touch must not overlap.
/// - `shape`, `src_strides` and `dst_strides` have the same length.
///
/// # Errors
///
/// Returns `Error::Internal` when the rank exceeds [`MAX_DIMS`] or the
/// launch fails, and `Error::InvalidArgument` when the element count exceeds
/// the 32-bit index of the kernel.
pub unsafe fn launch_strided_copy_into(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    src_ptr: u64,
    dst_ptr: u64,
    shape: &[usize],
    src_strides: &[isize],
    dst_strides: &[isize],
    elem_size: usize,
    src_byte_offset: usize,
    dst_byte_offset: usize,
) -> Result<()> {
    let ndim = shape.len();
    let numel: usize = shape.iter().product();
    if numel == 0 || ndim == 0 {
        return Ok(());
    }
    if ndim > MAX_DIMS {
        return Err(Error::Internal(format!(
            "strided_copy_into supports at most {MAX_DIMS} dimensions, got {ndim}"
        )));
    }
    let numel_u32 = u32::try_from(numel).map_err(|_| Error::InvalidArgument {
        arg: "numel",
        reason: format!("{numel} elements exceed the 32-bit index of strided_copy_into"),
    })?;

    let mut shape_args = [0u64; MAX_DIMS];
    let mut src_stride_args = [0i64; MAX_DIMS];
    let mut dst_stride_args = [0i64; MAX_DIMS];
    for d in 0..ndim {
        shape_args[d] = shape[d] as u64;
        src_stride_args[d] = src_strides[d] as i64;
        dst_stride_args[d] = dst_strides[d] as i64;
    }

    // SAFETY: the caller guarantees the pointers, offsets and strides stay in
    // bounds. Argument order and types match `strided_copy_into` in the .cu file.
    unsafe {
        let module = get_or_load_module(context, device_index, STRIDED_COPY_INTO_MODULE)?;
        let func = get_kernel_function(&module, "strided_copy_into")?;

        let grid = elementwise_launch_config(numel)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let elem_size_u32 = elem_size as u32;
        let src_offset_u64 = src_byte_offset as u64;
        let dst_offset_u64 = dst_byte_offset as u64;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&dst_ptr);
        for arg in &shape_args {
            builder.arg(arg);
        }
        for arg in &src_stride_args {
            builder.arg(arg);
        }
        for arg in &dst_stride_args {
            builder.arg(arg);
        }
        builder.arg(&numel_u32);
        builder.arg(&ndim_u32);
        builder.arg(&elem_size_u32);
        builder.arg(&src_offset_u64);
        builder.arg(&dst_offset_u64);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA strided_copy_into kernel launch failed: {:?}",
                e
            ))
        })?;
    }
    Ok(())
}
