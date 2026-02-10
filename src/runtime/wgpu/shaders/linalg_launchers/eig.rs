//! Eigendecomposition: symmetric (Jacobi) and general (Schur-based)

use wgpu::{Buffer, Queue};

use super::check_dtype_f32;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::wgpu::shaders::linalg_shaders::eig_general::EIG_GENERAL_SHADER;
use crate::runtime::wgpu::shaders::linalg_shaders::eig_symmetric::EIG_SYMMETRIC_SHADER;
use crate::runtime::wgpu::shaders::linalg_shaders::schur::SCHUR_SHADER;
use crate::runtime::wgpu::shaders::pipeline::{LayoutKey, PipelineCache};

/// Launch eigendecomposition kernel using Jacobi algorithm for symmetric matrices.
///
/// # Arguments
/// * `work` - Working matrix buffer `[n * n]`, will be diagonalized
/// * `eigenvectors` - Eigenvector matrix buffer `[n * n]`, stores column eigenvectors
/// * `eigenvalues` - Eigenvalue vector buffer `[n]`
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

    let module = cache.get_or_create_module("linalg_eig_symmetric", EIG_SYMMETRIC_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "linalg_eig_symmetric",
        "eig_jacobi_symmetric_f32",
        &module,
        &layout,
    );

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

    let module = cache.get_or_create_module("linalg_schur", SCHUR_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg_schur", "schur_decompose_f32", &module, &layout);

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

/// Launch rsf2csf kernel to convert Real Schur Form to Complex Schur Form.
///
/// Processes 2x2 blocks representing complex conjugate eigenvalue pairs and
/// converts them to upper triangular form with complex eigenvalues on diagonal.
///
/// # Arguments
/// * `t_real` - Real part of T matrix buffer [n * n], input/output
/// * `t_imag` - Imaginary part of T matrix buffer [n * n], output
/// * `z_real` - Real part of Z matrix buffer [n * n], input/output
/// * `z_imag` - Imaginary part of Z matrix buffer [n * n], output
/// * `params_buffer` - Parameters buffer with n (matrix dimension)
pub fn launch_rsf2csf(
    cache: &PipelineCache,
    queue: &Queue,
    t_real: &Buffer,
    t_imag: &Buffer,
    z_real: &Buffer,
    z_imag: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "rsf2csf");

    let module = cache.get_or_create_module("linalg_schur", SCHUR_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("linalg_schur", "rsf2csf_f32", &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[t_real, t_imag, z_real, z_imag, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rsf2csf"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rsf2csf"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single thread for sequential algorithm
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch QZ decomposition kernel for generalized Schur decomposition.
///
/// Computes Q, Z, S, T such that Q^T @ A @ Z = S (quasi-triangular)
/// and Q^T @ B @ Z = T (upper triangular).
///
/// # Arguments
/// * `s` - Matrix A buffer `[n * n]`, outputs S (quasi-triangular Schur form)
/// * `t` - Matrix B buffer `[n * n]`, outputs T (upper triangular)
/// * `q` - Left orthogonal matrix buffer `[n * n]`
/// * `z` - Right orthogonal matrix buffer `[n * n]`
/// * `eval_real` - Real parts of generalized eigenvalues buffer `[n]`
/// * `eval_imag` - Imaginary parts of generalized eigenvalues buffer `[n]`
/// * `converged_flag` - Convergence flag buffer (atomic i32): 0 if converged, 1 if not
/// * `params_buffer` - Parameters buffer with n (matrix dimension)
pub fn launch_qz_decompose(
    cache: &PipelineCache,
    queue: &Queue,
    s: &Buffer,
    t: &Buffer,
    q: &Buffer,
    z: &Buffer,
    eval_real: &Buffer,
    eval_imag: &Buffer,
    converged_flag: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "qz_decompose");

    let module = cache.get_or_create_module("linalg_eig_general", EIG_GENERAL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 7,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg_eig_general", "qz_decompose_f32", &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            s,
            t,
            q,
            z,
            eval_real,
            eval_imag,
            converged_flag,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("qz_decompose"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("qz_decompose"),
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
/// * `t` - Working matrix buffer `[n * n]`, modified during computation
/// * `z` - Schur transformation matrix buffer `[n * n]`
/// * `eval_real` - Real parts of eigenvalues buffer `[n]`
/// * `eval_imag` - Imaginary parts of eigenvalues buffer `[n]`
/// * `evec_real` - Real parts of eigenvectors buffer `[n * n]`
/// * `evec_imag` - Imaginary parts of eigenvectors buffer `[n * n]`
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

    let module = cache.get_or_create_module("linalg_eig_general", EIG_GENERAL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 7,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg_eig_general", "eig_general_f32", &module, &layout);

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
