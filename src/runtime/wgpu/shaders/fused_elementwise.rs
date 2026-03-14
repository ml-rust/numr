//! Fused elementwise WGSL kernel launchers. F32 only.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

const TERNARY_SHADER: &str = include_str!("fused_elementwise.wgsl");
const SCALAR_SHADER: &str = include_str!("fused_elementwise_scalar.wgsl");

/// Params for ternary ops (matches TernaryParams in WGSL)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TernaryParams {
    numel: u32,
}

/// Params for scalar FMA (matches ScalarFmaParams in WGSL)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalarFmaParams {
    numel: u32,
    scale: f32,
    bias: f32,
    _pad: u32,
}

fn launch_ternary(
    cache: &PipelineCache,
    queue: &Queue,
    entry_point: &'static str,
    op_name: &'static str,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    out: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op: op_name });
    }

    let params = TernaryParams {
        numel: numel as u32,
    };
    let params_buf = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("fused_elem_params"),
        size: std::mem::size_of::<TernaryParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let module = cache.get_or_create_module("fused_elementwise_f32", TERNARY_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("fused_elementwise_f32", entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[a, b, c, out, &params_buf]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(op_name),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op_name),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch fused_mul_add: out = a * b + c. F32 only.
pub fn launch_fused_mul_add(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    out: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    launch_ternary(
        cache,
        queue,
        "fused_mul_add_f32",
        "fused_mul_add",
        a,
        b,
        c,
        out,
        numel,
        dtype,
    )
}

/// Launch fused_add_mul: out = (a + b) * c. F32 only.
pub fn launch_fused_add_mul(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    out: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    launch_ternary(
        cache,
        queue,
        "fused_add_mul_f32",
        "fused_add_mul",
        a,
        b,
        c,
        out,
        numel,
        dtype,
    )
}

/// Launch fused_mul_add_scalar: out = a * scale + bias. F32 only.
pub fn launch_fused_mul_add_scalar(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    out: &Buffer,
    numel: usize,
    dtype: DType,
    scale: f32,
    bias: f32,
) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "fused_mul_add_scalar",
        });
    }

    let params = ScalarFmaParams {
        numel: numel as u32,
        scale,
        bias,
        _pad: 0,
    };
    let params_buf = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("fused_elem_scalar_params"),
        size: std::mem::size_of::<ScalarFmaParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let module = cache.get_or_create_module("fused_elementwise_scalar_f32", SCALAR_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "fused_elementwise_scalar_f32",
        "fused_mul_add_scalar_f32",
        &module,
        &layout,
    );
    let bind_group = cache.create_bind_group(&layout, &[a, out, &params_buf]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fused_mul_add_scalar"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_mul_add_scalar"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
