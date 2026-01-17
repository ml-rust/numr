//! Binary operation CUDA kernel launchers
//!
//! Provides launchers for element-wise binary operations (add, sub, mul, div, etc.)
//! on two tensors of the same shape.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_names,
    launch_binary_kernel, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch a binary operation kernel.
///
/// Performs element-wise operation: `output[i] = op(a[i], b[i])`
///
/// # Supported Operations
///
/// - `add`: Element-wise addition
/// - `sub`: Element-wise subtraction
/// - `mul`: Element-wise multiplication
/// - `div`: Element-wise division
/// - `pow`: Element-wise power
/// - `max`: Element-wise maximum
/// - `min`: Element-wise minimum
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have at least `numel` elements
/// - `a` and `b` must have the same dtype
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `op` - Operation name (e.g., "add", "mul")
/// * `dtype` - Data type of the tensors
/// * `a_ptr` - Device pointer to first input tensor
/// * `b_ptr` - Device pointer to second input tensor
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_binary_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_binary_kernel(
            context,
            stream,
            device_index,
            kernel_names::BINARY_MODULE,
            op,
            dtype,
            a_ptr,
            b_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch a logical_and kernel.
///
/// Performs element-wise logical AND: `output[i] = a[i] && b[i]`
/// All tensors are U8 (boolean: 0 = false, non-zero = true).
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have at least `numel` U8 elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `a_ptr` - Device pointer to first input tensor (U8)
/// * `b_ptr` - Device pointer to second input tensor (U8)
/// * `out_ptr` - Device pointer to output tensor (U8)
/// * `numel` - Number of elements
pub unsafe fn launch_logical_and_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::BINARY_MODULE)?;
        let func_name = "logical_and_u8";
        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA logical_and kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch a logical_or kernel.
///
/// Performs element-wise logical OR: `output[i] = a[i] || b[i]`
/// All tensors are U8 (boolean: 0 = false, non-zero = true).
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have at least `numel` U8 elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `a_ptr` - Device pointer to first input tensor (U8)
/// * `b_ptr` - Device pointer to second input tensor (U8)
/// * `out_ptr` - Device pointer to output tensor (U8)
/// * `numel` - Number of elements
pub unsafe fn launch_logical_or_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::BINARY_MODULE)?;
        let func_name = "logical_or_u8";
        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA logical_or kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch a logical_xor kernel.
///
/// Performs element-wise logical XOR: `output[i] = a[i] ^ b[i]`
/// All tensors are U8 (boolean: 0 = false, non-zero = true).
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have at least `numel` U8 elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `a_ptr` - Device pointer to first input tensor (U8)
/// * `b_ptr` - Device pointer to second input tensor (U8)
/// * `out_ptr` - Device pointer to output tensor (U8)
/// * `numel` - Number of elements
pub unsafe fn launch_logical_xor_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::BINARY_MODULE)?;
        let func_name = "logical_xor_u8";
        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA logical_xor kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}
