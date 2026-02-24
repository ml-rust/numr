//! Fused activation-mul WGSL kernel launchers. F32 only.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

const FUSED_ACTIVATION_MUL_SHADER: &str = include_str!("fused_activation_mul.wgsl");

// ============================================================================
// Forward launchers: (a, b) -> out
// ============================================================================

fn launch_fused_fwd(
    cache: &PipelineCache,
    queue: &Queue,
    entry_point: &'static str,
    op_name: &'static str,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op: op_name });
    }

    let module =
        cache.get_or_create_module("fused_activation_mul_f32", FUSED_ACTIVATION_MUL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("fused_activation_mul_f32", entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[a, b, out, params_buffer]);

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

/// Launch fused SiLU-mul forward: `out = silu(a) * b`. F32 only.
pub fn launch_silu_mul(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    launch_fused_fwd(
        cache,
        queue,
        "silu_mul_f32",
        "silu_mul",
        a,
        b,
        out,
        params_buffer,
        numel,
        dtype,
    )
}

/// Launch fused GELU-mul forward: `out = gelu(a) * b`. F32 only.
pub fn launch_gelu_mul(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    launch_fused_fwd(
        cache,
        queue,
        "gelu_mul_f32",
        "gelu_mul",
        a,
        b,
        out,
        params_buffer,
        numel,
        dtype,
    )
}

/// Launch fused ReLU-mul forward: `out = relu(a) * b`. F32 only.
pub fn launch_relu_mul(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    launch_fused_fwd(
        cache,
        queue,
        "relu_mul_f32",
        "relu_mul",
        a,
        b,
        out,
        params_buffer,
        numel,
        dtype,
    )
}

/// Launch fused sigmoid-mul forward: `out = sigmoid(a) * b`. F32 only.
pub fn launch_sigmoid_mul(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    launch_fused_fwd(
        cache,
        queue,
        "sigmoid_mul_f32",
        "sigmoid_mul",
        a,
        b,
        out,
        params_buffer,
        numel,
        dtype,
    )
}

// ============================================================================
// Backward launchers: (grad, a, b) -> (d_a, d_b)
// ============================================================================

fn launch_fused_bwd(
    cache: &PipelineCache,
    queue: &Queue,
    entry_point: &'static str,
    op_name: &'static str,
    grad: &Buffer,
    a: &Buffer,
    b: &Buffer,
    d_a: &Buffer,
    d_b: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op: op_name });
    }

    let module =
        cache.get_or_create_module("fused_activation_mul_f32", FUSED_ACTIVATION_MUL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("fused_activation_mul_f32", entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[grad, a, b, d_a, d_b, params_buffer]);

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

/// Launch fused SiLU-mul backward. F32 only.
pub fn launch_silu_mul_bwd(
    cache: &PipelineCache,
    queue: &Queue,
    grad: &Buffer,
    a: &Buffer,
    b: &Buffer,
    d_a: &Buffer,
    d_b: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    launch_fused_bwd(
        cache,
        queue,
        "silu_mul_bwd_f32",
        "silu_mul_bwd",
        grad,
        a,
        b,
        d_a,
        d_b,
        params_buffer,
        numel,
        dtype,
    )
}

/// Launch fused GELU-mul backward. F32 only.
pub fn launch_gelu_mul_bwd(
    cache: &PipelineCache,
    queue: &Queue,
    grad: &Buffer,
    a: &Buffer,
    b: &Buffer,
    d_a: &Buffer,
    d_b: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    launch_fused_bwd(
        cache,
        queue,
        "gelu_mul_bwd_f32",
        "gelu_mul_bwd",
        grad,
        a,
        b,
        d_a,
        d_b,
        params_buffer,
        numel,
        dtype,
    )
}

/// Launch fused ReLU-mul backward. F32 only.
pub fn launch_relu_mul_bwd(
    cache: &PipelineCache,
    queue: &Queue,
    grad: &Buffer,
    a: &Buffer,
    b: &Buffer,
    d_a: &Buffer,
    d_b: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    launch_fused_bwd(
        cache,
        queue,
        "relu_mul_bwd_f32",
        "relu_mul_bwd",
        grad,
        a,
        b,
        d_a,
        d_b,
        params_buffer,
        numel,
        dtype,
    )
}

/// Launch fused sigmoid-mul backward. F32 only.
pub fn launch_sigmoid_mul_bwd(
    cache: &PipelineCache,
    queue: &Queue,
    grad: &Buffer,
    a: &Buffer,
    b: &Buffer,
    d_a: &Buffer,
    d_b: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    launch_fused_bwd(
        cache,
        queue,
        "sigmoid_mul_bwd_f32",
        "sigmoid_mul_bwd",
        grad,
        a,
        b,
        d_a,
        d_b,
        params_buffer,
        numel,
        dtype,
    )
}
