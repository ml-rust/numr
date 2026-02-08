//! Triangular system solvers: forward and backward substitution

use wgpu::{Buffer, Queue};

use super::check_dtype_f32;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::wgpu::shaders::linalg_shaders::solvers::SOLVERS_SHADER;
use crate::runtime::wgpu::shaders::pipeline::{LayoutKey, PipelineCache};

/// Launch forward substitution kernel to solve Lx = b.
pub fn launch_forward_sub(
    cache: &PipelineCache,
    queue: &Queue,
    l: &Buffer,
    b: &Buffer,
    x: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "forward_sub");

    let module = cache.get_or_create_module("linalg_solvers", SOLVERS_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg_solvers", "forward_sub_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[l, b, x, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_sub"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("forward_sub"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single thread for sequential algorithm
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch backward substitution kernel to solve Ux = b.
pub fn launch_backward_sub(
    cache: &PipelineCache,
    queue: &Queue,
    u: &Buffer,
    b: &Buffer,
    x: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "backward_sub");

    let module = cache.get_or_create_module("linalg_solvers", SOLVERS_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg_solvers", "backward_sub_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[u, b, x, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("backward_sub"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("backward_sub"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single thread for sequential algorithm
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
