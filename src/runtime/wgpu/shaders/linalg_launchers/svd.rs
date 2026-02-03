//! Singular Value Decomposition (One-Sided Jacobi algorithm)

use wgpu::{Buffer, Queue};

use super::check_dtype_f32;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::wgpu::shaders::linalg_shaders::svd::SVD_SHADER;
use crate::runtime::wgpu::shaders::pipeline::{LayoutKey, PipelineCache};

/// Launch SVD decomposition kernel using One-Sided Jacobi algorithm.
///
/// # Arguments
/// * `b` - Working matrix buffer [m * n], will contain U columns after kernel completes
/// * `v` - V matrix buffer [n * n], will contain V (not transposed) after kernel completes
/// * `s` - Singular values buffer [n]
/// * `converged_flag` - Convergence flag buffer (atomic i32): 0 if converged, 1 if not
/// * `params_buffer` - Parameters buffer with (work_m, work_n)
pub fn launch_svd_jacobi(
    cache: &PipelineCache,
    queue: &Queue,
    b: &Buffer,
    v: &Buffer,
    s: &Buffer,
    converged_flag: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "svd_jacobi");

    let module = cache.get_or_create_module("linalg_svd", SVD_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg_svd", "svd_jacobi_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[b, v, s, converged_flag, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("svd_jacobi"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("svd_jacobi"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single thread for sequential algorithm
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
