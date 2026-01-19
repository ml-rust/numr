//! Binary operation CUDA kernel launchers
//!
//! Provides launchers for element-wise binary operations (add, sub, mul, div, etc.)
//! on two tensors of the same shape.
//!
//! Also supports broadcasting operations using strided access patterns.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    kernel_names, launch_binary_kernel, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::{CudaDevice, CudaRuntime};
use crate::tensor::Tensor;

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

/// Compute broadcast strides for a tensor shape relative to the output shape.
///
/// For each dimension in the output shape:
/// - If the input dimension matches, use the original stride
/// - If the input dimension is 1 (broadcast), use stride 0
/// - If the input doesn't have this dimension (prepended), use stride 0
pub fn compute_broadcast_strides(input_shape: &[usize], output_shape: &[usize]) -> Vec<u32> {
    let mut strides = vec![0u32; output_shape.len()];
    let input_ndim = input_shape.len();
    let output_ndim = output_shape.len();

    // Compute input strides (row-major)
    let mut input_strides = vec![1usize; input_ndim];
    for i in (0..input_ndim.saturating_sub(1)).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // Map input dimensions to output dimensions (right-aligned)
    let offset = output_ndim - input_ndim;
    for i in 0..output_ndim {
        if i < offset {
            // Dimension doesn't exist in input, broadcast with stride 0
            strides[i] = 0;
        } else {
            let input_idx = i - offset;
            if input_shape[input_idx] == 1 {
                // Broadcasting dimension, stride 0
                strides[i] = 0;
            } else {
                // Normal dimension, use input stride
                strides[i] = input_strides[input_idx] as u32;
            }
        }
    }

    strides
}

/// Launch a broadcast binary operation kernel.
///
/// Performs element-wise operation with broadcasting: `output[i] = op(a[broadcast_idx], b[broadcast_idx])`
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
/// - Shape arrays must be valid
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `op` - Operation name (e.g., "add", "mul")
/// * `dtype` - Data type of the tensors
/// * `device` - CUDA device for tensor allocation
/// * `a_ptr` - Device pointer to first input tensor
/// * `b_ptr` - Device pointer to second input tensor
/// * `out_ptr` - Device pointer to output tensor
/// * `a_shape` - Shape of tensor a
/// * `b_shape` - Shape of tensor b
/// * `out_shape` - Shape of output tensor (broadcast result)
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_broadcast_binary_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &CudaDevice,
    op: &str,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Result<()> {
    let numel: usize = out_shape.iter().product();
    if numel == 0 {
        return Ok(());
    }

    let ndim = out_shape.len();

    // Compute broadcast strides
    let a_strides = compute_broadcast_strides(a_shape, out_shape);
    let b_strides = compute_broadcast_strides(b_shape, out_shape);
    let out_strides: Vec<u32> = {
        let mut s = vec![1u32; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * out_shape[i + 1] as u32;
        }
        s
    };
    let shape_u32: Vec<u32> = out_shape.iter().map(|&x| x as u32).collect();

    // Allocate device memory for strides and shape using Tensor
    let a_strides_tensor = Tensor::<CudaRuntime>::from_slice(&a_strides, &[ndim], device);
    let b_strides_tensor = Tensor::<CudaRuntime>::from_slice(&b_strides, &[ndim], device);
    let out_strides_tensor = Tensor::<CudaRuntime>::from_slice(&out_strides, &[ndim], device);
    let shape_tensor = Tensor::<CudaRuntime>::from_slice(&shape_u32, &[ndim], device);

    // Get device pointers
    let a_strides_ptr = a_strides_tensor.storage().ptr();
    let b_strides_ptr = b_strides_tensor.storage().ptr();
    let out_strides_ptr = out_strides_tensor.storage().ptr();
    let shape_ptr = shape_tensor.storage().ptr();

    // Get kernel function
    let module = get_or_load_module(context, device_index, kernel_names::BINARY_MODULE)?;
    let func_name = format!(
        "{}_broadcast_{}",
        op,
        kernel_name("", dtype).trim_start_matches('_')
    );
    let func = get_kernel_function(&module, &func_name)?;

    // Launch kernel
    let grid = elementwise_launch_config(numel);
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let ndim_u32 = ndim as u32;

    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        builder.arg(&a_strides_ptr);
        builder.arg(&b_strides_ptr);
        builder.arg(&out_strides_ptr);
        builder.arg(&shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA broadcast binary kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    // Synchronize to ensure the kernel completes before freeing temporary allocations
    stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("Stream sync failed: {:?}", e)))?;

    Ok(())
}
