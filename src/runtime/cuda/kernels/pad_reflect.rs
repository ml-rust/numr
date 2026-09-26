//! Reflect-mode pad launcher: gathers every output element from a mirrored
//! source coordinate. See `pad_reflect.cu` for the per-element algorithm.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;

/// Module name for the reflect-pad kernels.
pub const PAD_REFLECT_MODULE: &str = "pad_reflect";

/// Maximum number of tensor dimensions supported. Must match `SHAPE_MAX_DIMS`
/// in `pad_reflect.cu`.
const MAX_DIMS: usize = 8;

/// Launch the reflect-mode pad kernel.
///
/// Shape/pad arrays are passed as individual scalar kernel arguments (not
/// device pointers) so this launcher is safe for CUDA graph capture/replay.
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - GPU device index
/// * `dtype` - Data type of the tensors
/// * `src_ptr` - Pointer to source tensor data (must be contiguous)
/// * `dst_ptr` - Pointer to output tensor data
/// * `src_shape` - Shape of source tensor
/// * `out_shape` - Shape of output tensor
/// * `pad_before` - Padding before each dimension
///
/// # Errors
///
/// Returns `Error::BackendLimitation` if `src_shape.len() > MAX_DIMS`.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Caller must have already validated every pad size is strictly less than
///   its dimension's size (see `validate_reflect_pad`)
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_pad_reflect(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    src_ptr: u64,
    dst_ptr: u64,
    src_shape: &[usize],
    out_shape: &[usize],
    pad_before: &[usize],
) -> Result<()> {
    let total_elements: usize = out_shape.iter().product();
    if total_elements == 0 {
        return Ok(());
    }

    let ndim = src_shape.len();
    if ndim > MAX_DIMS {
        return Err(Error::BackendLimitation {
            backend: "CUDA",
            operation: "pad_reflect",
            reason: format!(
                "tensor has {} dimensions but pad_reflect kernel supports at most {}",
                ndim, MAX_DIMS
            ),
        });
    }

    let mut src_shape_args = [0u32; MAX_DIMS];
    let mut out_shape_args = [0u32; MAX_DIMS];
    let mut pad_before_args = [0u32; MAX_DIMS];

    for i in 0..ndim {
        src_shape_args[i] = src_shape[i] as u32;
        out_shape_args[i] = out_shape[i] as u32;
        pad_before_args[i] = pad_before[i] as u32;
    }

    unsafe {
        let module = get_or_load_module(context, device_index, PAD_REFLECT_MODULE)?;
        let func_name = kernel_name("pad_reflect", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total_elements)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let total_u32 = total_elements as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&dst_ptr);
        for i in 0..MAX_DIMS {
            builder.arg(&src_shape_args[i]);
        }
        for i in 0..MAX_DIMS {
            builder.arg(&out_shape_args[i]);
        }
        for i in 0..MAX_DIMS {
            builder.arg(&pad_before_args[i]);
        }
        builder.arg(&ndim_u32);
        builder.arg(&total_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA {func_name} kernel launch failed: {e:?}"))
        })?;

        Ok(())
    }
}
