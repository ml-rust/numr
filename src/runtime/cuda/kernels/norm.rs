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

/// Launch a GroupNorm kernel.
///
/// Computes: Group normalization across divided channel groups
/// Input shape: [batch, channels, spatial...]
/// Divides channels into num_groups, normalizes each group separately
///
/// Computes for each (batch, group):
/// - mean and variance over channels_per_group * spatial elements
/// - Then: `output = (input - mean) / sqrt(variance + eps) * weight + bias`
///
/// # Arguments
///
/// * `input_ptr` - Device pointer to input tensor of shape [batch, channels, spatial...]
/// * `weight_ptr` - Device pointer to weight tensor of shape [channels]
/// * `bias_ptr` - Device pointer to bias tensor of shape [channels]
/// * `output_ptr` - Device pointer to output tensor of shape [batch, channels, spatial...]
/// * `batch` - Batch size
/// * `channels` - Number of channels
/// * `spatial` - Product of spatial dimensions (height * width for 4D tensors)
/// * `num_groups` - Number of groups to divide channels into
/// * `channels_per_group` - Channels per group (channels / num_groups)
/// * `eps` - Small constant for numerical stability (typically 1e-5)
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Input and output must have batch * channels * spatial elements
/// - Weight and bias must have channels elements
/// - channels must be divisible by num_groups
pub unsafe fn launch_group_norm(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    weight_ptr: u64,
    bias_ptr: u64,
    output_ptr: u64,
    batch: usize,
    channels: usize,
    spatial: usize,
    num_groups: usize,
    channels_per_group: usize,
    eps: f32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::NORM_MODULE)?;
        let func_name = kernel_name("group_norm", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        // One block per (batch, group) pair
        let grid_size = (batch * num_groups) as u32;
        let group_size = channels_per_group * spatial;
        let block_size = BLOCK_SIZE.min(group_size as u32);

        // Shared memory: 2 * block_size floats for mean and variance reduction
        let shared_mem = block_size * 2 * 4; // 2 floats per thread for f32

        let batch_u32 = batch as u32;
        let channels_u32 = channels as u32;
        let spatial_u32 = spatial as u32;
        let num_groups_u32 = num_groups as u32;
        let channels_per_group_u32 = channels_per_group as u32;

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), shared_mem);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&output_ptr);
        builder.arg(&batch_u32);
        builder.arg(&channels_u32);
        builder.arg(&spatial_u32);
        builder.arg(&num_groups_u32);
        builder.arg(&channels_per_group_u32);
        builder.arg(&eps);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA group_norm kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}
