//! Ternary operation CUDA kernel launchers
//!
//! Provides launchers for ternary operations like where (conditional select).
//! where(cond, x, y) = cond ? x : y

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    kernel_names, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch a where (conditional select) kernel.
///
/// Performs element-wise conditional selection: `output[i] = cond[i] ? x[i] : y[i]`
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have at least `numel` elements
/// - Condition tensor must be U8 (boolean: 0 = false, non-zero = true)
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of x, y, and output tensors
/// * `cond_ptr` - Device pointer to condition tensor (U8)
/// * `x_ptr` - Device pointer to "true" values tensor
/// * `y_ptr` - Device pointer to "false" values tensor
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_where_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    cond_ptr: u64,
    x_ptr: u64,
    y_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::TERNARY_MODULE)?;
        let func_name = kernel_name("where", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&cond_ptr);
        builder.arg(&x_ptr);
        builder.arg(&y_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA where kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;

        Ok(())
    }
}

/// Launch a broadcast where (conditional select) kernel.
///
/// Performs element-wise conditional selection with broadcasting support:
/// `output[i] = cond[cond_offset] ? x[x_offset] : y[y_offset]`
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Stride and shape arrays must have `ndim` elements
/// - Output tensor must have `numel` elements
/// - Condition tensor must be U8 (boolean: 0 = false, non-zero = true)
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of x, y, and output tensors
/// * `cond_ptr` - Device pointer to condition tensor (U8)
/// * `x_ptr` - Device pointer to "true" values tensor
/// * `y_ptr` - Device pointer to "false" values tensor
/// * `out_ptr` - Device pointer to output tensor
/// * `cond_strides` - Device pointer to condition tensor strides
/// * `x_strides` - Device pointer to x tensor strides
/// * `y_strides` - Device pointer to y tensor strides
/// * `shape` - Device pointer to output shape
/// * `ndim` - Number of dimensions
/// * `numel` - Number of output elements
// Currently unused: CUDA where_cond uses CPU fallback for broadcasting
// This is available for future optimization of GPU-native broadcasting
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_where_broadcast_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    cond_ptr: u64,
    x_ptr: u64,
    y_ptr: u64,
    out_ptr: u64,
    cond_strides: u64,
    x_strides: u64,
    y_strides: u64,
    shape: u64,
    ndim: usize,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::TERNARY_MODULE)?;
        let func_name = format!("where_broadcast_{}", super::loader::dtype_suffix(dtype));
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;
        let ndim_u32 = ndim as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&cond_ptr);
        builder.arg(&x_ptr);
        builder.arg(&y_ptr);
        builder.arg(&out_ptr);
        builder.arg(&cond_strides);
        builder.arg(&x_strides);
        builder.arg(&y_strides);
        builder.arg(&shape);
        builder.arg(&ndim_u32);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA where_broadcast kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;

        Ok(())
    }
}
