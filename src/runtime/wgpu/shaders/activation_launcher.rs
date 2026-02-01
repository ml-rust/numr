//! Activation and utility WGSL kernel launchers
//!
//! Provides launchers for specialized activation and utility operations:
//! - `launch_leaky_relu` - Leaky ReLU activation
//! - `launch_elu` - ELU (Exponential Linear Unit) activation
//! - `launch_clamp_op` - Value clamping

use wgpu::{Buffer, Queue};

use super::elementwise_wgsl::ELEMENTWISE_SHADER;
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Parametric Activation Operations
// ============================================================================

/// Launch Leaky ReLU activation kernel.
///
/// Computes `out[i] = max(negative_slope * a[i], a[i])` for all elements.
///
/// Helps prevent "dying ReLU" by allowing small gradients for negative inputs.
pub fn launch_leaky_relu(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    // leaky_relu is float-only
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "leaky_relu",
        });
    }

    let module = cache.get_or_create_module("elementwise", ELEMENTWISE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("elementwise", "leaky_relu_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("leaky_relu"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("leaky_relu"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch ELU (Exponential Linear Unit) activation kernel.
///
/// Computes `out[i] = a[i] if a[i] > 0, else alpha * (exp(a[i]) - 1)` for all elements.
///
/// Smooth approximation to ReLU with negative values saturating to -alpha.
pub fn launch_elu(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    // elu is float-only
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op: "elu" });
    }

    let module = cache.get_or_create_module("elementwise", ELEMENTWISE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("elementwise", "elu_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("elu") });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("elu"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Clamp Operation
// ============================================================================

/// Launch clamp operation kernel.
///
/// Computes `out[i] = clamp(a[i], min_val, max_val)` for all elements.
pub fn launch_clamp_op(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    // clamp is float-only
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op: "clamp" });
    }

    let module = cache.get_or_create_module("elementwise", ELEMENTWISE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("elementwise", "clamp_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("clamp"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("clamp"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
