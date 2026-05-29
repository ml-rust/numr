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
use crate::runtime::cuda::CudaDevice;

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

/// Maximum number of dimensions supported by the inline broadcast kernel.
///
/// Must match `MAX_BROADCAST_DIMS` in `binary.cu`.
pub const MAX_BROADCAST_DIMS: usize = 8;

/// Launch a broadcast binary operation kernel.
///
/// Performs element-wise operation with broadcasting:
/// `output[i] = op(a[broadcast_idx], b[broadcast_idx])`
///
/// # CUDA Graph Compatibility
///
/// This function uses the `*_broadcast_*_inline` kernel variants that accept
/// strides and shape as individual scalar u32 arguments baked into the
/// kernel-parameter block.  Unlike the pointer-based variants, the inline
/// kernels do NOT trigger H2D memcpy nodes during CUDA graph capture, so the
/// graph's kernel nodes never contain stale host-side pointers.
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
/// - `out_shape.len()` must be ≤ `MAX_BROADCAST_DIMS` (= 8)
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
/// * `a_shape` - Shape of tensor a
/// * `b_shape` - Shape of tensor b
/// * `out_shape` - Shape of output tensor (broadcast result)
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_broadcast_binary_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    _device: &CudaDevice,
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
    if ndim > MAX_BROADCAST_DIMS {
        return Err(Error::Internal(format!(
            "launch_broadcast_binary_op: ndim={ndim} exceeds MAX_BROADCAST_DIMS={MAX_BROADCAST_DIMS}"
        )));
    }

    // Compute broadcast strides.
    let a_strides_vec = compute_broadcast_strides(a_shape, out_shape);
    let b_strides_vec = compute_broadcast_strides(b_shape, out_shape);
    let shape_vec: Vec<u32> = out_shape.iter().map(|&x| x as u32).collect();

    // Pack into fixed-size arrays (zero-padded to MAX_BROADCAST_DIMS).
    // These arrays live in Rust stack memory and are passed as individual
    // u32 scalar arguments to the kernel — no H2D copy is performed, making
    // this safe inside a CUDA graph capture region.
    let mut a_strides = [0u32; MAX_BROADCAST_DIMS];
    let mut b_strides = [0u32; MAX_BROADCAST_DIMS];
    let mut shape = [0u32; MAX_BROADCAST_DIMS];
    for i in 0..ndim {
        a_strides[i] = a_strides_vec[i];
        b_strides[i] = b_strides_vec[i];
        shape[i] = shape_vec[i];
    }

    // Select the inline kernel variant (no device pointers for strides/shape).
    let module = get_or_load_module(context, device_index, kernel_names::BINARY_MODULE)?;
    let func_name = format!(
        "{}_broadcast_{}_inline",
        op,
        kernel_name("", dtype).trim_start_matches('_')
    );
    let func = get_kernel_function(&module, &func_name)?;

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
        // a_strides[0..7] as individual args
        builder.arg(&a_strides[0]);
        builder.arg(&a_strides[1]);
        builder.arg(&a_strides[2]);
        builder.arg(&a_strides[3]);
        builder.arg(&a_strides[4]);
        builder.arg(&a_strides[5]);
        builder.arg(&a_strides[6]);
        builder.arg(&a_strides[7]);
        // b_strides[0..7] as individual args
        builder.arg(&b_strides[0]);
        builder.arg(&b_strides[1]);
        builder.arg(&b_strides[2]);
        builder.arg(&b_strides[3]);
        builder.arg(&b_strides[4]);
        builder.arg(&b_strides[5]);
        builder.arg(&b_strides[6]);
        builder.arg(&b_strides[7]);
        // shape[0..7] as individual args
        builder.arg(&shape[0]);
        builder.arg(&shape[1]);
        builder.arg(&shape[2]);
        builder.arg(&shape[3]);
        builder.arg(&shape[4]);
        builder.arg(&shape[5]);
        builder.arg(&shape[6]);
        builder.arg(&shape[7]);
        builder.arg(&ndim_u32);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA broadcast binary kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}
