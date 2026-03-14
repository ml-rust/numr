//! Activation and utility WGSL kernel launchers. F32 only.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

const SCALAR_SHADER: &str = include_str!("scalar.wgsl");
const ACTIVATION_SHADER: &str = include_str!("activation.wgsl");

/// Launch Leaky ReLU: `out[i] = max(slope * a[i], a[i])`. F32 only.
pub fn launch_leaky_relu(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "leaky_relu",
        });
    }

    let module = cache.get_or_create_module("scalar_f32", SCALAR_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("scalar_f32", "leaky_relu_f32", &module, &layout);
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

/// Launch ELU: `out[i] = x > 0 ? x : alpha * (exp(x) - 1)`. F32 only.
pub fn launch_elu(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op: "elu" });
    }

    let module = cache.get_or_create_module("scalar_f32", SCALAR_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("scalar_f32", "elu_f32", &module, &layout);
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

/// Launch clamp: `out[i] = clamp(a[i], min_val, max_val)`. F32 only.
pub fn launch_clamp_op(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op: "clamp" });
    }

    let module = cache.get_or_create_module("activation_f32", ACTIVATION_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("activation_f32", "clamp_f32", &module, &layout);
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
