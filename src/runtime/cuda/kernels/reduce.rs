//! Reduction CUDA kernel launchers
//!
//! Provides launchers for reduction operations (sum, max, min) that reduce
//! tensors along specified dimensions.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    get_kernel_function, get_or_load_module, kernel_name, kernel_names, launch_config,
    reduce_dim_launch_config, reduce_launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch a global reduction kernel.
///
/// Performs a parallel reduction across all elements, producing partial results
/// (one per block). For complete reduction, call multiple times until only one
/// element remains.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `input_ptr` must have at least `numel` elements
/// - `output_ptr` must have space for the number of blocks launched
///
/// # Returns
///
/// The number of blocks launched (equals the number of partial results).
#[allow(dead_code)] // Kept for potential future optimization of global reductions
pub unsafe fn launch_reduce_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<u32> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::REDUCE_MODULE)?;
        let func_name = kernel_name(&kernel_names::reduce_kernel(op), dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid_size, block_size) = reduce_launch_config(numel);
        let n = numel as u32;

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA reduce kernel '{}' launch failed: {:?}",
                op, e
            ))
        })?;

        Ok(grid_size)
    }
}

/// Launch a dimension-wise reduction kernel.
///
/// Reduces a tensor along a single dimension, preserving the outer and inner
/// dimensions. The tensor is conceptually reshaped to `[outer, reduce, inner]`
/// and reduced along the middle dimension.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `input_ptr` must have `outer_size * reduce_size * inner_size` elements
/// - `output_ptr` must have `outer_size * inner_size` elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `op` - Reduction operation ("sum", "max", or "min")
/// * `dtype` - Data type of the tensor
/// * `input_ptr` - Device pointer to input tensor
/// * `output_ptr` - Device pointer to output tensor
/// * `outer_size` - Product of dimensions before the reduction dimension
/// * `reduce_size` - Size of the dimension being reduced
/// * `inner_size` - Product of dimensions after the reduction dimension
pub unsafe fn launch_reduce_dim_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::REDUCE_MODULE)?;
        let func_name = kernel_name(&kernel_names::reduce_dim_kernel(op), dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid, block) = reduce_dim_launch_config(outer_size, inner_size);
        let outer = outer_size as u32;
        let reduce = reduce_size as u32;
        let inner = inner_size as u32;

        let cfg = launch_config(grid, (block, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer);
        builder.arg(&reduce);
        builder.arg(&inner);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA reduce_dim kernel '{}' launch failed: {:?}",
                op, e
            ))
        })?;

        Ok(())
    }
}
