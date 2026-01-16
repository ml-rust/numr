//! Normalization CUDA kernel launchers
//!
//! Provides launchers for normalization operations (RMSNorm, LayerNorm)
//! commonly used in transformer architectures.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, get_kernel_function, get_or_load_module, kernel_name, kernel_names, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Calculate launch configuration for normalization kernels.
///
/// One block per row (batch element), with threads cooperating to compute statistics.
/// Returns (grid_size, block_size, shared_memory_bytes).
#[inline]
fn norm_launch_config(batch_size: usize, hidden_size: usize) -> (u32, u32, u32) {
    let block_size = BLOCK_SIZE.min(hidden_size as u32);
    let grid_size = batch_size as u32;
    // Shared memory: block_size floats for reduction
    // For layer_norm we need 2x for mean and variance
    let shared_mem = block_size * 4; // f32
    (grid_size, block_size, shared_mem)
}

/// Launch a RMSNorm (Root Mean Square Layer Normalization) kernel.
///
/// Computes: `output = input * rsqrt(mean(input^2) + eps) * weight`
///
/// RMSNorm is used in LLaMA and other modern transformer architectures.
/// It's simpler and faster than LayerNorm as it doesn't require computing mean.
///
/// # Arguments
///
/// * `input_ptr` - Device pointer to input tensor of shape [batch_size, hidden_size]
/// * `weight_ptr` - Device pointer to weight tensor of shape [hidden_size]
/// * `output_ptr` - Device pointer to output tensor of shape [batch_size, hidden_size]
/// * `batch_size` - Number of rows (batch dimension)
/// * `hidden_size` - Size of each row (hidden dimension)
/// * `eps` - Small constant for numerical stability (typically 1e-5 or 1e-6)
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `input_ptr` and `output_ptr` must have `batch_size * hidden_size` elements
/// - `weight_ptr` must have `hidden_size` elements
pub unsafe fn launch_rms_norm(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    weight_ptr: u64,
    output_ptr: u64,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::NORM_MODULE)?;
        let func_name = kernel_name("rms_norm", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid_size, block_size, shared_mem) = norm_launch_config(batch_size, hidden_size);
        let batch = batch_size as u32;
        let hidden = hidden_size as u32;

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), shared_mem);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&output_ptr);
        builder.arg(&batch);
        builder.arg(&hidden);
        builder.arg(&eps);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA rms_norm kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}

/// Launch a LayerNorm kernel.
///
/// Computes: `output = (input - mean) / sqrt(variance + eps) * weight + bias`
///
/// LayerNorm normalizes across the hidden dimension for each batch element.
///
/// # Arguments
///
/// * `input_ptr` - Device pointer to input tensor of shape [batch_size, hidden_size]
/// * `weight_ptr` - Device pointer to weight (gamma) tensor of shape [hidden_size]
/// * `bias_ptr` - Device pointer to bias (beta) tensor of shape [hidden_size]
/// * `output_ptr` - Device pointer to output tensor of shape [batch_size, hidden_size]
/// * `batch_size` - Number of rows (batch dimension)
/// * `hidden_size` - Size of each row (hidden dimension)
/// * `eps` - Small constant for numerical stability (typically 1e-5)
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `input_ptr` and `output_ptr` must have `batch_size * hidden_size` elements
/// - `weight_ptr` and `bias_ptr` must have `hidden_size` elements
pub unsafe fn launch_layer_norm(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    weight_ptr: u64,
    bias_ptr: u64,
    output_ptr: u64,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::NORM_MODULE)?;
        let func_name = kernel_name("layer_norm", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid_size, block_size, shared_mem) = norm_launch_config(batch_size, hidden_size);
        // Layer norm needs 2x shared memory for mean and variance
        let shared_mem = shared_mem * 2;
        let batch = batch_size as u32;
        let hidden = hidden_size as u32;

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), shared_mem);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&output_ptr);
        builder.arg(&batch);
        builder.arg(&hidden);
        builder.arg(&eps);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA layer_norm kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}
