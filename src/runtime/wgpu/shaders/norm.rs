//! Normalization WGSL kernel launchers
//!
//! Provides launchers for normalization operations:
//! - RMS normalization
//! - Layer normalization
//!
//! All operations run entirely on GPU with no CPU fallback.

use wgpu::{Buffer, Queue};

use super::norm_wgsl::NORM_SHADER;
use super::pipeline::{LayoutKey, PipelineCache};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Helper Macros
// ============================================================================

macro_rules! check_dtype_f32 {
    ($dtype:expr, $op:expr) => {
        if $dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype: $dtype,
                op: $op,
            });
        }
    };
}

// ============================================================================
// RMS Normalization
// ============================================================================

/// Launch RMS normalization kernel.
///
/// Computes: output = input / sqrt(mean(input^2) + eps) * weight
pub fn launch_rms_norm(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    weight: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "rms_norm");

    let module = cache.get_or_create_module("norm", NORM_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("norm", "rms_norm_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, weight, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rms_norm"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rms_norm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per batch element
        pass.dispatch_workgroups(batch_size as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Layer Normalization
// ============================================================================

/// Launch layer normalization kernel with bias.
///
/// Computes: output = (input - mean) / sqrt(var + eps) * weight + bias
pub fn launch_layer_norm(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    weight: &Buffer,
    bias: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "layer_norm");

    let module = cache.get_or_create_module("norm", NORM_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("norm", "layer_norm_f32", &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[input, weight, bias, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("layer_norm"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("layer_norm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per batch element
        pass.dispatch_workgroups(batch_size as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch layer normalization kernel without bias.
///
/// Computes: output = (input - mean) / sqrt(var + eps) * weight
pub fn launch_layer_norm_no_bias(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    weight: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "layer_norm_no_bias");

    let module = cache.get_or_create_module("norm", NORM_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("norm", "layer_norm_no_bias_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, weight, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("layer_norm_no_bias"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("layer_norm_no_bias"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per batch element
        pass.dispatch_workgroups(batch_size as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
