//! Banded solver kernel launchers: Thomas algorithm and banded LU

use wgpu::{Buffer, Queue};

use super::check_dtype_f32;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::wgpu::shaders::linalg_shaders::banded::BANDED_SHADER;
use crate::runtime::wgpu::shaders::pipeline::{LayoutKey, PipelineCache};

/// Launch Thomas solver for tridiagonal systems (kl=1, ku=1).
pub fn launch_thomas_solve(
    cache: &PipelineCache,
    queue: &Queue,
    ab: &Buffer,
    b: &Buffer,
    x: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "thomas_solve");

    let module = cache.get_or_create_module("linalg_banded", BANDED_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg_banded", "thomas_solve_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[ab, b, x, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("thomas_solve"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("thomas_solve"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single thread for sequential algorithm
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch banded LU solver for general banded systems.
pub fn launch_banded_lu_solve(
    cache: &PipelineCache,
    queue: &Queue,
    ab: &Buffer,
    b: &Buffer,
    x: &Buffer,
    work: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "banded_lu_solve");

    let module = cache.get_or_create_module("linalg_banded", BANDED_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg_banded", "banded_lu_solve_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[ab, b, x, work, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("banded_lu_solve"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("banded_lu_solve"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single thread for sequential algorithm
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
