//! Eigendecomposition: symmetric (Jacobi) and general (Schur-based)

use wgpu::{Buffer, Queue};

use super::check_dtype_f32;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::wgpu::shaders::linalg_wgsl::LINALG_SHADER;
use crate::runtime::wgpu::shaders::pipeline::{LayoutKey, PipelineCache};

/// Launch eigendecomposition kernel using Jacobi algorithm for symmetric matrices.
///
/// # Arguments
/// * `work` - Working matrix buffer [n * n], will be diagonalized
/// * `eigenvectors` - Eigenvector matrix buffer [n * n], stores column eigenvectors
/// * `eigenvalues` - Eigenvalue vector buffer [n]
/// * `converged_flag` - Convergence flag buffer (atomic i32): 0 if converged, 1 if not
/// * `params_buffer` - Parameters buffer with n (matrix dimension)
pub fn launch_eig_jacobi_symmetric(
    cache: &PipelineCache,
    queue: &Queue,
    work: &Buffer,
    eigenvectors: &Buffer,
    eigenvalues: &Buffer,
    converged_flag: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "eig_jacobi_symmetric");

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg", "eig_jacobi_symmetric_f32", &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            work,
            eigenvectors,
            eigenvalues,
            converged_flag,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("eig_jacobi_symmetric"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("eig_jacobi_symmetric"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single thread for sequential algorithm
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch Schur decomposition kernel for general (non-symmetric) matrices.
///
/// Computes A = Z @ T @ Z^T where T is quasi-upper-triangular (real Schur form)
/// and Z is orthogonal.
///
/// # Arguments
/// * `t` - Matrix buffer [n * n], contains input A, outputs T (quasi-triangular)
/// * `z` - Orthogonal transformation matrix buffer [n * n]
/// * `converged_flag` - Convergence flag buffer (atomic i32): 0 if converged, 1 if not
/// * `params_buffer` - Parameters buffer with n (matrix dimension)
pub fn launch_schur_decompose(
    cache: &PipelineCache,
    queue: &Queue,
    t: &Buffer,
    z: &Buffer,
    converged_flag: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "schur_decompose");

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "schur_decompose_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[t, z, converged_flag, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("schur_decompose"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("schur_decompose"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single thread for sequential algorithm
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch general eigenvalue decomposition kernel for non-symmetric matrices.
///
/// Uses Schur decomposition + back-substitution to compute eigenvalues and eigenvectors.
/// Returns real and imaginary parts of eigenvalues and eigenvectors.
///
/// # Arguments
/// * `t` - Working matrix buffer [n * n], modified during computation
/// * `z` - Schur transformation matrix buffer [n * n]
/// * `eval_real` - Real parts of eigenvalues buffer [n]
/// * `eval_imag` - Imaginary parts of eigenvalues buffer [n]
/// * `evec_real` - Real parts of eigenvectors buffer [n * n]
/// * `evec_imag` - Imaginary parts of eigenvectors buffer [n * n]
/// * `converged_flag` - Convergence flag buffer (atomic i32): 0 if converged, 1 if not
/// * `params_buffer` - Parameters buffer with n (matrix dimension)
pub fn launch_eig_general(
    cache: &PipelineCache,
    queue: &Queue,
    t: &Buffer,
    z: &Buffer,
    eval_real: &Buffer,
    eval_imag: &Buffer,
    evec_real: &Buffer,
    evec_imag: &Buffer,
    converged_flag: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "eig_general");

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 7,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "eig_general_f32", &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            t,
            z,
            eval_real,
            eval_imag,
            evec_real,
            evec_imag,
            converged_flag,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("eig_general"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("eig_general"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single thread for sequential algorithm
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
