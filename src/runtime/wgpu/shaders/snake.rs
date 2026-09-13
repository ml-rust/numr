//! Snake activation WGSL kernel launchers. F32 only.
//!
//! Kernel source: snake.wgsl

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

const SNAKE_SHADER: &str = include_str!("snake.wgsl");
const SNAKE_MODULE: &str = "snake_f32";

/// Uniform block shared by every snake entry point. Matches `SnakeParams` in
/// snake.wgsl: `extent` is `numel` for the elementwise entry points and
/// `outer` for the per-channel reduction.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SnakeParams {
    /// `numel` for the elementwise entry points, `outer` for the reduction.
    pub extent: u32,
    /// Length of the channel axis.
    pub channels: u32,
    /// Product of the dims after the channel axis.
    pub inner: u32,
    /// Denominator guard added to `beta`.
    pub eps: f32,
}

fn dispatch(
    cache: &PipelineCache,
    queue: &Queue,
    entry_point: &'static str,
    buffers: &[&Buffer],
    workgroups: u32,
    dtype: DType,
) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: entry_point,
        });
    }
    let module = cache.get_or_create_module(SNAKE_MODULE, SNAKE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: (buffers.len() - 1) as u32,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(SNAKE_MODULE, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, buffers);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(entry_point),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(entry_point),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// `out = x + sin(alpha * x)^2 / (beta + eps)` over `numel` elements.
#[allow(clippy::too_many_arguments)]
pub fn launch_snake_beta(
    cache: &PipelineCache,
    queue: &Queue,
    x: &Buffer,
    alpha: &Buffer,
    beta: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    dispatch(
        cache,
        queue,
        "snake_beta_f32",
        &[x, alpha, beta, out, params_buffer],
        workgroup_count(numel),
        dtype,
    )
}

/// `d_x = grad * (1 + alpha * sin(2 alpha x) / (beta + eps))` over `numel` elements.
#[allow(clippy::too_many_arguments)]
pub fn launch_snake_beta_dx(
    cache: &PipelineCache,
    queue: &Queue,
    grad: &Buffer,
    x: &Buffer,
    alpha: &Buffer,
    beta: &Buffer,
    d_x: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    dispatch(
        cache,
        queue,
        "snake_beta_dx_f32",
        &[grad, x, alpha, beta, d_x, params_buffer],
        workgroup_count(numel),
        dtype,
    )
}

/// Per-channel `d_alpha` / `d_beta` sums, one workgroup per channel.
#[allow(clippy::too_many_arguments)]
pub fn launch_snake_beta_dparams(
    cache: &PipelineCache,
    queue: &Queue,
    grad: &Buffer,
    x: &Buffer,
    alpha: &Buffer,
    beta: &Buffer,
    d_alpha: &Buffer,
    d_beta: &Buffer,
    params_buffer: &Buffer,
    channels: usize,
    dtype: DType,
) -> Result<()> {
    dispatch(
        cache,
        queue,
        "snake_beta_dparams_f32",
        &[grad, x, alpha, beta, d_alpha, d_beta, params_buffer],
        channels as u32,
        dtype,
    )
}
