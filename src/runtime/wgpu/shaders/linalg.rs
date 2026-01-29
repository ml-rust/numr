//! Linear algebra WGSL kernel launchers
//!
//! Provides launchers for linear algebra operations including:
//! - Matrix decompositions (LU, Cholesky, QR)
//! - Triangular solvers (forward/backward substitution)
//! - Matrix operations (trace, diag, diagflat, determinant)
//!
//! Follows the same API pattern as CUDA kernel launchers for DRY consistency.

use wgpu::{Buffer, Queue};

use super::linalg_wgsl::LINALG_SHADER;
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Helper Macros
// ============================================================================

macro_rules! check_dtype_f32 {
    ($dtype:expr, $op:expr) => {
        if $dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype: $dtype,
                op: $op,
            });
        }
    };
}

// ============================================================================
// Trace - Sum of diagonal elements
// ============================================================================

/// Launch trace kernel to compute sum of diagonal elements.
pub fn launch_trace(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "trace");

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "trace_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("trace"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("trace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Diag - Extract diagonal elements
// ============================================================================

/// Launch diag kernel to extract diagonal elements from matrix.
pub fn launch_diag(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    min_dim: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "diag");

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "diag_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("diag"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("diag"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(min_dim), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Diagflat - Create diagonal matrix from vector
// ============================================================================

/// Launch diagflat kernel to create diagonal matrix from vector.
pub fn launch_diagflat(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "diagflat");

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "diagflat_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("diagflat"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("diagflat"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n * n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Create Identity Matrix
// ============================================================================

/// Create identity matrix on GPU.
pub fn launch_create_identity(
    cache: &PipelineCache,
    queue: &Queue,
    output: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "create_identity");

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "create_identity_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("create_identity"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("create_identity"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n * n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Forward Substitution - Solve Lx = b
// ============================================================================

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

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "forward_sub_f32", &module, &layout);

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

// ============================================================================
// Backward Substitution - Solve Ux = b
// ============================================================================

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

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "backward_sub_f32", &module, &layout);

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

// ============================================================================
// LU Decomposition
// ============================================================================

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

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "lu_decompose_f32", &module, &layout);

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

// ============================================================================
// Cholesky Decomposition
// ============================================================================

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

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
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

// ============================================================================
// QR Decomposition
// ============================================================================

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

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "qr_decompose_f32", &module, &layout);

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

// ============================================================================
// Determinant from LU
// ============================================================================

/// Launch determinant computation from LU decomposition.
pub fn launch_det_from_lu(
    cache: &PipelineCache,
    queue: &Queue,
    lu_matrix: &Buffer,
    det_output: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "det_from_lu");

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "det_from_lu_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[lu_matrix, det_output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("det_from_lu"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("det_from_lu"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Apply LU Permutation
// ============================================================================

/// Apply LU permutation to a vector.
pub fn launch_apply_lu_permutation(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    pivots: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "apply_lu_permutation");

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg", "apply_lu_permutation_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, output, pivots, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apply_lu_permutation"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apply_lu_permutation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Scatter Column
// ============================================================================

/// Scatter vector into a column of a matrix.
pub fn launch_scatter_column(
    cache: &PipelineCache,
    queue: &Queue,
    vec: &Buffer,
    matrix: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "scatter_column");

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "scatter_column_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[vec, matrix, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scatter_column"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter_column"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Extract Column
// ============================================================================

/// Extract a column from a matrix.
pub fn launch_extract_column(
    cache: &PipelineCache,
    queue: &Queue,
    matrix: &Buffer,
    col_out: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "extract_column");

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "extract_column_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[matrix, col_out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("extract_column"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("extract_column"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(m), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Max Absolute Value
// ============================================================================

/// Compute maximum absolute value of elements.
pub fn launch_max_abs(
    cache: &PipelineCache,
    queue: &Queue,
    values: &Buffer,
    max_output: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "max_abs");

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "max_abs_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[values, max_output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("max_abs"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("max_abs"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Count Above Threshold
// ============================================================================

/// Count elements with absolute value above threshold.
pub fn launch_count_above_threshold(
    cache: &PipelineCache,
    queue: &Queue,
    values: &Buffer,
    count_output: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "count_above_threshold");

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg", "count_above_threshold_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[values, count_output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("count_above_threshold"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("count_above_threshold"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Matrix Copy
// ============================================================================

/// Copy matrix data on device.
pub fn launch_matrix_copy(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "matrix_copy");

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "matrix_copy_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, dst, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matrix_copy"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matrix_copy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// SVD Decomposition (One-Sided Jacobi)
// ============================================================================

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

    let module = cache.get_or_create_module("linalg", LINALG_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("linalg", "svd_jacobi_f32", &module, &layout);

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

// ============================================================================
// Eigendecomposition (Jacobi Algorithm for Symmetric Matrices)
// ============================================================================

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

// ============================================================================
// Schur Decomposition (Hessenberg + QR Iteration)
// ============================================================================

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

// ============================================================================
// General Eigendecomposition (for non-symmetric matrices)
// ============================================================================

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
