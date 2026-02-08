//! Activation and utility WGSL kernel launchers
//!
//! Provides launchers for specialized activation and utility operations:
//! - `launch_leaky_relu` - Leaky ReLU activation
//! - `launch_elu` - ELU (Exponential Linear Unit) activation
//! - `launch_clamp_op` - Value clamping
//!
//! All operations support F32 and F16 dtypes.

use wgpu::{Buffer, Queue};

use super::generator::{
    dtype_suffix, generate_clamp_shader, generate_scalar_shader, is_wgsl_float,
};
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
///
/// Supports F32 and F16 dtypes.
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
    if !is_wgsl_float(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "leaky_relu",
        });
    }

    let suffix = dtype_suffix(dtype)?;
    let shader_key = format!("scalar_{}", suffix);
    let entry_point = format!("leaky_relu_{}", suffix);

    let shader_source = generate_scalar_shader(dtype)?;
    let module = cache.get_or_create_module_from_source(&shader_key, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_dynamic_pipeline(&shader_key, &entry_point, &module, &layout);

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
///
/// Supports F32 and F16 dtypes.
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
    if !is_wgsl_float(dtype) {
        return Err(Error::UnsupportedDType { dtype, op: "elu" });
    }

    let suffix = dtype_suffix(dtype)?;
    let shader_key = format!("scalar_{}", suffix);
    let entry_point = format!("elu_{}", suffix);

    let shader_source = generate_scalar_shader(dtype)?;
    let module = cache.get_or_create_module_from_source(&shader_key, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_dynamic_pipeline(&shader_key, &entry_point, &module, &layout);

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
///
/// Supports F32 and F16 dtypes.
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
    if !is_wgsl_float(dtype) {
        return Err(Error::UnsupportedDType { dtype, op: "clamp" });
    }

    let suffix = dtype_suffix(dtype)?;
    let shader_key = format!("clamp_{}", suffix);
    let entry_point = format!("clamp_{}", suffix);

    let shader_source = generate_clamp_shader(dtype)?;
    let module = cache.get_or_create_module_from_source(&shader_key, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_dynamic_pipeline(&shader_key, &entry_point, &module, &layout);

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
