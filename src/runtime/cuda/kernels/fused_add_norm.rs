//! Fused Add + Normalization CUDA kernel launchers
//!
//! Provides launchers for fused operations combining residual addition with normalization.
//! These operations are common in transformer architectures for efficient computation.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, get_kernel_function, get_or_load_module, kernel_name, kernel_names, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Calculate launch configuration for fused normalization kernels.
///
/// One block per row (batch element), with threads cooperating to compute statistics.
/// Returns (grid_size, block_size, shared_memory_bytes).
#[inline]
fn fused_norm_launch_config(
    batch_size: usize,
    hidden_size: usize,
    shared_arrays: usize,
    dtype: DType,
) -> (u32, u32, u32) {
    let block_size = BLOCK_SIZE.min(hidden_size as u32);
    let grid_size = batch_size as u32;
    let elem_size = match dtype {
        DType::F64 => 8u32,
        _ => 4u32, // f32, f16, bf16 all use f32 shared memory
    };
    let shared_mem = (shared_arrays as u32) * block_size * elem_size;
    (grid_size, block_size, shared_mem)
}

/// Launch a fused_add_rms_norm forward kernel.
///
/// Computes: `pre_norm = input + residual`, then `output = pre_norm * rsqrt(mean(pre_norm^2) + eps) * weight`
///
/// # Arguments
///
/// * `input_ptr` - Device pointer to input tensor of shape [batch_size, hidden_size]
/// * `residual_ptr` - Device pointer to residual tensor of shape [batch_size, hidden_size]
/// * `weight_ptr` - Device pointer to weight tensor of shape [hidden_size]
/// * `output_ptr` - Device pointer to output tensor of shape [batch_size, hidden_size]
/// * `pre_norm_ptr` - Device pointer to pre-normalization tensor of shape [batch_size, hidden_size]
/// * `batch_size` - Number of rows (batch dimension)
/// * `hidden_size` - Size of each row (hidden dimension)
/// * `eps` - Small constant for numerical stability
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have `batch_size * hidden_size` elements
pub unsafe fn launch_fused_add_rms_norm(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    residual_ptr: u64,
    weight_ptr: u64,
    output_ptr: u64,
    pre_norm_ptr: u64,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) -> Result<()> {
    unsafe {
        let module =
            get_or_load_module(context, device_index, kernel_names::FUSED_ADD_NORM_MODULE)?;
        let func_name = kernel_name("fused_add_rms_norm", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid_size, block_size, shared_mem) =
            fused_norm_launch_config(batch_size, hidden_size, 1, dtype);
        let batch = batch_size as u32;
        let hidden = hidden_size as u32;
        let eps_f64 = eps as f64;

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), shared_mem);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&residual_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&output_ptr);
        builder.arg(&pre_norm_ptr);
        builder.arg(&batch);
        builder.arg(&hidden);
        if dtype == DType::F64 {
            builder.arg(&eps_f64);
        } else {
            builder.arg(&eps);
        }

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA fused_add_rms_norm kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch a fused_add_rms_norm backward kernel.
///
/// Computes gradients for fused add + RMSNorm operation.
///
/// # Arguments
///
/// * `grad_ptr` - Device pointer to gradient tensor of shape [batch_size, hidden_size]
/// * `pre_norm_ptr` - Device pointer to pre-norm tensor from forward pass
/// * `weight_ptr` - Device pointer to weight tensor of shape [hidden_size]
/// * `d_input_residual_ptr` - Device pointer to output gradients for input and residual
/// * `d_weight_ptr` - Device pointer to weight gradients (pre-zeroed, accumulated via atomicAdd)
/// * `batch_size` - Number of rows
/// * `hidden_size` - Size of each row
/// * `eps` - Small constant for numerical stability
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - d_weight_ptr must be pre-zeroed with `hidden_size` elements
pub unsafe fn launch_fused_add_rms_norm_bwd(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    grad_ptr: u64,
    pre_norm_ptr: u64,
    weight_ptr: u64,
    d_input_residual_ptr: u64,
    d_weight_ptr: u64,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) -> Result<()> {
    unsafe {
        let module =
            get_or_load_module(context, device_index, kernel_names::FUSED_ADD_NORM_MODULE)?;
        let func_name = kernel_name("fused_add_rms_norm_bwd", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        // Backward needs 2 shared arrays: sum_sq and dot
        let (grid_size, block_size, shared_mem) =
            fused_norm_launch_config(batch_size, hidden_size, 2, dtype);
        let batch = batch_size as u32;
        let hidden = hidden_size as u32;
        let eps_f64 = eps as f64;

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), shared_mem);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&grad_ptr);
        builder.arg(&pre_norm_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&d_input_residual_ptr);
        builder.arg(&d_weight_ptr);
        builder.arg(&batch);
        builder.arg(&hidden);
        if dtype == DType::F64 {
            builder.arg(&eps_f64);
        } else {
            builder.arg(&eps);
        }

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA fused_add_rms_norm_bwd kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch a fused_add_layer_norm forward kernel.
///
/// Computes: `pre_norm = input + residual`, then
/// `output = (pre_norm - mean) / sqrt(var + eps) * weight + bias`
///
/// # Arguments
///
/// * `input_ptr` - Device pointer to input tensor of shape [batch_size, hidden_size]
/// * `residual_ptr` - Device pointer to residual tensor of shape [batch_size, hidden_size]
/// * `weight_ptr` - Device pointer to weight (gamma) tensor of shape [hidden_size]
/// * `bias_ptr` - Device pointer to bias (beta) tensor of shape [hidden_size]
/// * `output_ptr` - Device pointer to output tensor of shape [batch_size, hidden_size]
/// * `pre_norm_ptr` - Device pointer to pre-normalization tensor of shape [batch_size, hidden_size]
/// * `batch_size` - Number of rows (batch dimension)
/// * `hidden_size` - Size of each row (hidden dimension)
/// * `eps` - Small constant for numerical stability
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have `batch_size * hidden_size` elements
pub unsafe fn launch_fused_add_layer_norm(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    residual_ptr: u64,
    weight_ptr: u64,
    bias_ptr: u64,
    output_ptr: u64,
    pre_norm_ptr: u64,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) -> Result<()> {
    unsafe {
        let module =
            get_or_load_module(context, device_index, kernel_names::FUSED_ADD_NORM_MODULE)?;
        let func_name = kernel_name("fused_add_layer_norm", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        // Layer norm needs 2 shared arrays: mean and variance
        let (grid_size, block_size, shared_mem) =
            fused_norm_launch_config(batch_size, hidden_size, 2, dtype);
        let batch = batch_size as u32;
        let hidden = hidden_size as u32;
        let eps_f64 = eps as f64;

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), shared_mem);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&residual_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&output_ptr);
        builder.arg(&pre_norm_ptr);
        builder.arg(&batch);
        builder.arg(&hidden);
        if dtype == DType::F64 {
            builder.arg(&eps_f64);
        } else {
            builder.arg(&eps);
        }

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA fused_add_layer_norm kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch a fused_add_layer_norm backward kernel.
///
/// Computes gradients for fused add + LayerNorm operation.
///
/// # Arguments
///
/// * `grad_ptr` - Device pointer to gradient tensor of shape [batch_size, hidden_size]
/// * `pre_norm_ptr` - Device pointer to pre-norm tensor from forward pass
/// * `weight_ptr` - Device pointer to weight tensor of shape [hidden_size]
/// * `d_input_residual_ptr` - Device pointer to output gradients for input and residual
/// * `d_weight_ptr` - Device pointer to weight gradients (pre-zeroed, accumulated via atomicAdd)
/// * `d_bias_ptr` - Device pointer to bias gradients (pre-zeroed, accumulated via atomicAdd)
/// * `batch_size` - Number of rows
/// * `hidden_size` - Size of each row
/// * `eps` - Small constant for numerical stability
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - d_weight_ptr and d_bias_ptr must be pre-zeroed with `hidden_size` elements each
pub unsafe fn launch_fused_add_layer_norm_bwd(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    grad_ptr: u64,
    pre_norm_ptr: u64,
    weight_ptr: u64,
    d_input_residual_ptr: u64,
    d_weight_ptr: u64,
    d_bias_ptr: u64,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) -> Result<()> {
    unsafe {
        let module =
            get_or_load_module(context, device_index, kernel_names::FUSED_ADD_NORM_MODULE)?;
        let func_name = kernel_name("fused_add_layer_norm_bwd", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        // Backward needs 4 shared arrays: mean, var, gs (mean_gs), gsn (mean_gsn)
        let (grid_size, block_size, shared_mem) =
            fused_norm_launch_config(batch_size, hidden_size, 4, dtype);
        let batch = batch_size as u32;
        let hidden = hidden_size as u32;
        let eps_f64 = eps as f64;

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), shared_mem);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&grad_ptr);
        builder.arg(&pre_norm_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&d_input_residual_ptr);
        builder.arg(&d_weight_ptr);
        builder.arg(&d_bias_ptr);
        builder.arg(&batch);
        builder.arg(&hidden);
        if dtype == DType::F64 {
            builder.arg(&eps_f64);
        } else {
            builder.arg(&eps);
        }

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA fused_add_layer_norm_bwd kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}
