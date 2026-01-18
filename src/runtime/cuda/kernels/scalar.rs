//! Scalar operation CUDA kernel launchers
//!
//! Provides launchers for element-wise tensor-scalar operations
//! (add_scalar, mul_scalar, etc.).

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    kernel_names, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch a scalar operation kernel for f32.
///
/// Performs element-wise operation: `output[i] = op(input[i], scalar)`
///
/// # Supported Operations
///
/// - `add_scalar`: Add scalar to each element
/// - `sub_scalar`: Subtract scalar from each element
/// - `mul_scalar`: Multiply each element by scalar
/// - `div_scalar`: Divide each element by scalar
/// - `pow_scalar`: Raise each element to scalar power
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Tensors must have at least `numel` elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `op` - Operation name (e.g., "add_scalar", "mul_scalar")
/// * `a_ptr` - Device pointer to input tensor
/// * `scalar` - Scalar value
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_scalar_op_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    a_ptr: u64,
    scalar: f32,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SCALAR_MODULE)?;
        let func_name = kernel_name(op, DType::F32);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&scalar);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA scalar kernel '{}' launch failed: {:?}",
                op, e
            ))
        })?;

        Ok(())
    }
}

/// Launch a scalar operation kernel for f64.
///
/// See [`launch_scalar_op_f32`] for documentation.
///
/// # Safety
///
/// Same requirements as `launch_scalar_op_f32`.
pub unsafe fn launch_scalar_op_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    a_ptr: u64,
    scalar: f64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SCALAR_MODULE)?;
        let func_name = kernel_name(op, DType::F64);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&scalar);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA scalar kernel '{}' launch failed: {:?}",
                op, e
            ))
        })?;

        Ok(())
    }
}

/// Launch a scalar operation kernel for i32.
///
/// # Safety
///
/// Same requirements as `launch_scalar_op_f32`.
pub unsafe fn launch_scalar_op_i32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    a_ptr: u64,
    scalar: i32,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SCALAR_MODULE)?;
        let func_name = kernel_name(op, DType::I32);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&scalar);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA scalar kernel '{}' launch failed: {:?}",
                op, e
            ))
        })?;

        Ok(())
    }
}

/// Launch a scalar operation kernel for i64.
///
/// # Safety
///
/// Same requirements as `launch_scalar_op_f32`.
pub unsafe fn launch_scalar_op_i64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    a_ptr: u64,
    scalar: i64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SCALAR_MODULE)?;
        let func_name = kernel_name(op, DType::I64);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&scalar);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA scalar kernel '{}' launch failed: {:?}",
                op, e
            ))
        })?;

        Ok(())
    }
}

/// Launch a scalar operation kernel for f16/bf16/fp8 (uses f32 scalar value).
///
/// # Safety
///
/// Same requirements as `launch_scalar_op_f32`.
#[cfg(any(feature = "f16", feature = "fp8"))]
pub unsafe fn launch_scalar_op_half(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    dtype: DType,
    a_ptr: u64,
    scalar: f32,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SCALAR_MODULE)?;
        let func_name = kernel_name(op, dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&scalar);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA scalar kernel '{}' launch failed: {:?}",
                op, e
            ))
        })?;

        Ok(())
    }
}
