//! Activation function CUDA kernel launchers
//!
//! Provides launchers for activation functions (ReLU, sigmoid, softmax)
//! commonly used in neural networks.

use std::sync::Arc;
use cudarc::driver::safe::{CudaContext, CudaStream};
use cudarc::driver::PushKernelArg;

use super::loader::{
    get_kernel_function, get_or_load_module, kernel_name, kernel_names, launch_config,
    launch_unary_kernel, softmax_launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Element-wise Activations
// ============================================================================

/// Launch a ReLU (Rectified Linear Unit) kernel.
///
/// Computes: `output[i] = max(0, input[i])`
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Tensors must have at least `numel` elements
pub unsafe fn launch_relu(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> { unsafe {
    launch_unary_kernel(
        context,
        stream,
        device_index,
        kernel_names::ACTIVATION_MODULE,
        "relu",
        dtype,
        input_ptr,
        output_ptr,
        numel,
    )
}}

/// Launch a sigmoid kernel.
///
/// Computes: `output[i] = 1 / (1 + exp(-input[i]))`
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Tensors must have at least `numel` elements
pub unsafe fn launch_sigmoid(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> { unsafe {
    launch_unary_kernel(
        context,
        stream,
        device_index,
        kernel_names::ACTIVATION_MODULE,
        "sigmoid",
        dtype,
        input_ptr,
        output_ptr,
        numel,
    )
}}

// ============================================================================
// Softmax Activations
// ============================================================================

/// Launch softmax over the last dimension.
///
/// For a tensor of shape `[..., D]`, computes softmax independently for each
/// of the `outer_size` vectors of length `dim_size`.
///
/// The softmax is computed as:
/// ```text
/// softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
/// ```
///
/// Uses shared memory for parallel reduction of max and sum values.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `input_ptr` must have `outer_size * dim_size` elements
/// - `output_ptr` must have `outer_size * dim_size` elements
///
/// # Arguments
///
/// * `outer_size` - Number of independent softmax computations (product of all dims except last)
/// * `dim_size` - Size of the last dimension (the dimension over which softmax is computed)
pub unsafe fn launch_softmax(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    dim_size: usize,
) -> Result<()> { unsafe {
    let module = get_or_load_module(context, device_index, kernel_names::ACTIVATION_MODULE)?;
    let func_name = kernel_name("softmax", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let (grid_size, block_size, shared_mem) = softmax_launch_config(outer_size, dim_size);
    let outer = outer_size as u32;
    let dim = dim_size as u32;

    // Adjust shared memory for f64 (double the size)
    let shared_mem = if dtype == DType::F64 {
        shared_mem * 2
    } else {
        shared_mem
    };

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), shared_mem);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&outer);
    builder.arg(&dim);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("CUDA softmax kernel launch failed: {:?}", e)))?;

    Ok(())
}}

/// Launch softmax over a non-last dimension.
///
/// For a tensor of shape `[A, B, C]` with softmax over dimension 1:
/// - `outer_size` = A
/// - `dim_size` = B
/// - `inner_size` = C
///
/// Each thread handles one (outer, inner) position and sequentially computes
/// softmax across the `dim_size` elements.
///
/// # Performance Note
///
/// This kernel uses one thread per (outer, inner) position with sequential
/// processing over dim_size. For large dim_size values, consider using
/// `launch_softmax` by transposing the tensor to put the reduction dimension last.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `input_ptr` must have `outer_size * dim_size * inner_size` elements
/// - `output_ptr` must have `outer_size * dim_size * inner_size` elements
pub unsafe fn launch_softmax_dim(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
) -> Result<()> { unsafe {
    let module = get_or_load_module(context, device_index, kernel_names::ACTIVATION_MODULE)?;
    let func_name = kernel_name("softmax_dim", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    // The kernel uses blockIdx.x for outer and blockIdx.y for inner,
    // with each thread handling one (outer, inner) pair sequentially over dim_size.
    // This is intentionally a 2D grid with 1 thread per block to match the kernel design.
    let grid = (outer_size as u32, inner_size as u32, 1);
    let outer = outer_size as u32;
    let dim = dim_size as u32;
    let inner = inner_size as u32;

    let cfg = launch_config(grid, (1, 1, 1), 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&outer);
    builder.arg(&dim);
    builder.arg(&inner);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!("CUDA softmax_dim kernel launch failed: {:?}", e))
    })?;

    Ok(())
}}
