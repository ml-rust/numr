//! Statistics CUDA kernel launchers
//!
//! Provides launchers for statistics operations (mode) that run entirely on GPU
//! without CPU fallback.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    dtype_suffix, get_kernel_function, get_or_load_module, kernel_names, launch_config,
    reduce_dim_launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch mode_dim kernel for dimension-wise mode computation.
///
/// Finds the most frequent value along a dimension on sorted data.
/// The tensor is conceptually reshaped to `[outer, reduce, inner]` and
/// mode is computed along the middle dimension.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `sorted_ptr` must have `outer_size * reduce_size * inner_size` elements
/// - `mode_values_ptr` must have `outer_size * inner_size` elements
/// - `mode_counts_ptr` must have `outer_size * inner_size` I64 elements
/// - Input must be pre-sorted along the reduce dimension
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of the tensor
/// * `sorted_ptr` - Device pointer to sorted input tensor
/// * `mode_values_ptr` - Device pointer to output mode values
/// * `mode_counts_ptr` - Device pointer to output mode counts (I64)
/// * `outer_size` - Product of dimensions before the reduction dimension
/// * `reduce_size` - Size of the dimension being reduced
/// * `inner_size` - Product of dimensions after the reduction dimension
pub unsafe fn launch_mode_dim(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    sorted_ptr: u64,
    mode_values_ptr: u64,
    mode_counts_ptr: u64,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) -> Result<()> {
    let suffix = dtype_suffix(dtype);
    let func_name = format!("mode_dim_{}", suffix);

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::STATISTICS_MODULE)?;
        let func = get_kernel_function(&module, &func_name)?;

        // Each block handles one output element, thread 0 does the work
        let num_outputs = outer_size * inner_size;
        let (grid, block) = reduce_dim_launch_config(outer_size, inner_size);

        let outer = outer_size as u32;
        let reduce = reduce_size as u32;
        let inner = inner_size as u32;

        let cfg = launch_config(grid, (block, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&sorted_ptr);
        builder.arg(&mode_values_ptr);
        builder.arg(&mode_counts_ptr);
        builder.arg(&outer);
        builder.arg(&reduce);
        builder.arg(&inner);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA mode_dim kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;

        // Ensure we used num_outputs for sanity check
        let _ = num_outputs;

        Ok(())
    }
}

/// Launch mode_full kernel for full tensor mode computation.
///
/// Finds the most frequent value in the entire sorted tensor.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `sorted_ptr` must have `numel` elements
/// - `mode_value_ptr` must have 1 element
/// - `mode_count_ptr` must have 1 I64 element
/// - Input must be pre-sorted
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of the tensor
/// * `sorted_ptr` - Device pointer to sorted input tensor
/// * `mode_value_ptr` - Device pointer to output mode value (single element)
/// * `mode_count_ptr` - Device pointer to output mode count (single I64)
/// * `numel` - Total number of elements
pub unsafe fn launch_mode_full(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    sorted_ptr: u64,
    mode_value_ptr: u64,
    mode_count_ptr: u64,
    numel: usize,
) -> Result<()> {
    let suffix = dtype_suffix(dtype);
    let func_name = format!("mode_full_{}", suffix);

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::STATISTICS_MODULE)?;
        let func = get_kernel_function(&module, &func_name)?;

        // Single block, single thread
        let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
        let n = numel as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&sorted_ptr);
        builder.arg(&mode_value_ptr);
        builder.arg(&mode_count_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA mode_full kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;

        Ok(())
    }
}
