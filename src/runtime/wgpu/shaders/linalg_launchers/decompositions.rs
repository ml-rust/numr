//! Matrix decompositions: LU, Cholesky, QR

use wgpu::{Buffer, Queue};

use super::check_dtype_f32;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::wgpu::shaders::linalg_shaders::decompositions::DECOMPOSITIONS_SHADER;
use crate::runtime::wgpu::shaders::pipeline::{LayoutKey, PipelineCache};

/// Launch LU decomposition kernel with partial pivoting.
pub fn launch_lu_decompose(
    cache: &PipelineCache,
    queue: &Queue,
    lu_matrix: &Buffer,
    pivots: &Buffer,
    num_swaps: &Buffer,
    singular_flag: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "lu_decompose");

    let module = cache.get_or_create_module("linalg_decompositions", DECOMPOSITIONS_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(
        "linalg_decompositions",
        "lu_decompose_f32",
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(
        &layout,
        &[lu_matrix, pivots, num_swaps, singular_flag, params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lu_decompose"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lu_decompose"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single thread for sequential algorithm
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch Cholesky decomposition kernel.
pub fn launch_cholesky_decompose(
    cache: &PipelineCache,
    queue: &Queue,
    l_matrix: &Buffer,
    not_pd_flag: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "cholesky_decompose");

    let module = cache.get_or_create_module("linalg_decompositions", DECOMPOSITIONS_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg", "cholesky_decompose_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[l_matrix, not_pd_flag, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cholesky_decompose"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cholesky_decompose"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single thread for sequential algorithm
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch QR decomposition kernel using Householder reflections.
pub fn launch_qr_decompose(
    cache: &PipelineCache,
    queue: &Queue,
    q_matrix: &Buffer,
    r_matrix: &Buffer,
    workspace: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "qr_decompose");

    let module = cache.get_or_create_module("linalg_decompositions", DECOMPOSITIONS_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(
        "linalg_decompositions",
        "qr_decompose_f32",
        &module,
        &layout,
    );

    let bind_group =
        cache.create_bind_group(&layout, &[q_matrix, r_matrix, workspace, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("qr_decompose"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("qr_decompose"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single thread for sequential algorithm
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
