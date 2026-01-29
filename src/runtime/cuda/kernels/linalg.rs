//! Linear algebra CUDA kernel launchers
//!
//! Provides launchers for linear algebra operations including:
//! - Matrix decompositions (LU, Cholesky, QR)
//! - Triangular solvers (forward/backward substitution)
//! - Matrix operations (trace, diag, diagflat, determinant)

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_names,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Trace - Sum of diagonal elements
// ============================================================================

/// Launch trace kernel to compute sum of diagonal elements.
///
/// # Safety
///
/// - `input_ptr` must point to a valid [n, n] matrix
/// - `output_ptr` must point to a single element (will be atomically added to)
/// - Output should be zero-initialized before launch
pub unsafe fn launch_trace(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    n: usize,
    stride: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "trace_f32",
        DType::F64 => "trace_f64",
        _ => return Err(Error::UnsupportedDType { dtype, op: "trace" }),
    };

    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(n);
    let block = (BLOCK_SIZE, 1, 1);
    let n_u32 = n as u32;
    let stride_u32 = stride as u32;

    // Shared memory for reduction
    let shared_mem = BLOCK_SIZE * std::mem::size_of::<f64>() as u32;

    let cfg = launch_config(grid, block, shared_mem);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&n_u32);
    builder.arg(&stride_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA trace kernel launch failed: {:?}", e)))?;

    Ok(())
}

// ============================================================================
// Diag - Extract diagonal elements
// ============================================================================

/// Launch diag kernel to extract diagonal elements from matrix.
///
/// # Safety
///
/// - `input_ptr` must point to a valid [m, n] matrix
/// - `output_ptr` must have space for min(m, n) elements
pub unsafe fn launch_diag(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    min_dim: usize,
    n_cols: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "diag_f32",
        DType::F64 => "diag_f64",
        _ => return Err(Error::UnsupportedDType { dtype, op: "diag" }),
    };

    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(min_dim);
    let block = (BLOCK_SIZE, 1, 1);
    let min_dim_u32 = min_dim as u32;
    let n_cols_u32 = n_cols as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&min_dim_u32);
    builder.arg(&n_cols_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA diag kernel launch failed: {:?}", e)))?;

    Ok(())
}

// ============================================================================
// Diagflat - Create diagonal matrix from vector
// ============================================================================

/// Launch diagflat kernel to create diagonal matrix from vector.
///
/// # Safety
///
/// - `input_ptr` must point to a valid vector of n elements
/// - `output_ptr` must have space for n*n elements
pub unsafe fn launch_diagflat(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "diagflat_f32",
        DType::F64 => "diagflat_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "diagflat",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let total = n * n;
    let grid = elementwise_launch_config(total);
    let block = (BLOCK_SIZE, 1, 1);
    let n_u32 = n as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&n_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA diagflat kernel launch failed: {:?}", e)))?;

    Ok(())
}

// ============================================================================
// Forward Substitution - Solve Lx = b
// ============================================================================

/// Launch forward substitution kernel to solve Lx = b.
///
/// # Safety
///
/// - `l_ptr` must point to a valid [n, n] lower triangular matrix
/// - `b_ptr` must point to a valid vector of n elements
/// - `x_ptr` must have space for n elements
pub unsafe fn launch_forward_sub(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    l_ptr: u64,
    b_ptr: u64,
    x_ptr: u64,
    n: usize,
    unit_diagonal: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "forward_sub_f32",
        DType::F64 => "forward_sub_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "forward_sub",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    // Single thread kernel for sequential algorithm
    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;
    let unit_diag_i32 = if unit_diagonal { 1i32 } else { 0i32 };

    let mut builder = stream.launch_builder(&func);
    builder.arg(&l_ptr);
    builder.arg(&b_ptr);
    builder.arg(&x_ptr);
    builder.arg(&n_u32);
    builder.arg(&unit_diag_i32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA forward_sub kernel launch failed: {:?}", e)))?;

    Ok(())
}

// ============================================================================
// Backward Substitution - Solve Ux = b
// ============================================================================

/// Launch backward substitution kernel to solve Ux = b.
///
/// # Safety
///
/// - `u_ptr` must point to a valid [n, n] upper triangular matrix
/// - `b_ptr` must point to a valid vector of n elements
/// - `x_ptr` must have space for n elements
pub unsafe fn launch_backward_sub(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    u_ptr: u64,
    b_ptr: u64,
    x_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "backward_sub_f32",
        DType::F64 => "backward_sub_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "backward_sub",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    // Single thread kernel for sequential algorithm
    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&u_ptr);
    builder.arg(&b_ptr);
    builder.arg(&x_ptr);
    builder.arg(&n_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA backward_sub kernel launch failed: {:?}", e)))?;

    Ok(())
}

// ============================================================================
// LU Decomposition
// ============================================================================

/// Launch LU decomposition kernel with partial pivoting.
///
/// Modifies lu_ptr in-place to store L (below diagonal) and U (on/above diagonal).
///
/// # Safety
///
/// - `lu_ptr` must point to a valid [m, n] matrix (will be modified in-place)
/// - `pivots_ptr` must have space for min(m, n) i64 elements
/// - `num_swaps_ptr` must point to a single i32
/// - `singular_flag_ptr` must point to a single i32 (zero-initialized)
pub unsafe fn launch_lu_decompose(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    lu_ptr: u64,
    pivots_ptr: u64,
    num_swaps_ptr: u64,
    singular_flag_ptr: u64,
    m: usize,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "lu_decompose_f32",
        DType::F64 => "lu_decompose_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "lu_decompose",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    // Single thread kernel for sequential algorithm
    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let m_u32 = m as u32;
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&lu_ptr);
    builder.arg(&pivots_ptr);
    builder.arg(&num_swaps_ptr);
    builder.arg(&m_u32);
    builder.arg(&n_u32);
    builder.arg(&singular_flag_ptr);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA lu_decompose kernel launch failed: {:?}", e)))?;

    Ok(())
}

// ============================================================================
// Cholesky Decomposition
// ============================================================================

/// Launch Cholesky decomposition kernel.
///
/// Modifies l_ptr in-place to store L (lower triangular factor).
///
/// # Safety
///
/// - `l_ptr` must point to a valid [n, n] symmetric positive-definite matrix
/// - `not_pd_flag_ptr` must point to a single i32 (zero-initialized)
pub unsafe fn launch_cholesky_decompose(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    l_ptr: u64,
    not_pd_flag_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "cholesky_decompose_f32",
        DType::F64 => "cholesky_decompose_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "cholesky_decompose",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&l_ptr);
    builder.arg(&n_u32);
    builder.arg(&not_pd_flag_ptr);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA cholesky_decompose kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

// ============================================================================
// QR Decomposition
// ============================================================================

/// Launch QR decomposition kernel using Householder reflections.
///
/// # Safety
///
/// - `q_ptr` must have space for [m, m] (full) or [m, k] (thin) matrix
/// - `r_ptr` must point to a copy of input matrix [m, n] (modified in-place)
/// - `workspace_ptr` must have space for m elements (Householder vector)
pub unsafe fn launch_qr_decompose(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    q_ptr: u64,
    r_ptr: u64,
    workspace_ptr: u64,
    m: usize,
    n: usize,
    thin: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "qr_decompose_f32",
        DType::F64 => "qr_decompose_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "qr_decompose",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let thin_i32 = if thin { 1i32 } else { 0i32 };

    let mut builder = stream.launch_builder(&func);
    builder.arg(&q_ptr);
    builder.arg(&r_ptr);
    builder.arg(&workspace_ptr);
    builder.arg(&m_u32);
    builder.arg(&n_u32);
    builder.arg(&thin_i32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA qr_decompose kernel launch failed: {:?}", e)))?;

    Ok(())
}

// ============================================================================
// Determinant from LU
// ============================================================================

/// Launch determinant computation from LU decomposition.
///
/// # Safety
///
/// - `lu_ptr` must point to a valid [n, n] LU decomposition
/// - `det_ptr` must point to a single element
pub unsafe fn launch_det_from_lu(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    lu_ptr: u64,
    det_ptr: u64,
    n: usize,
    num_swaps: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "det_from_lu_f32",
        DType::F64 => "det_from_lu_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "det_from_lu",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&lu_ptr);
    builder.arg(&det_ptr);
    builder.arg(&n_u32);
    builder.arg(&num_swaps);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA det_from_lu kernel launch failed: {:?}", e)))?;

    Ok(())
}

// ============================================================================
// Apply LU Permutation
// ============================================================================

/// Apply LU permutation to a vector.
///
/// # Safety
///
/// - `in_ptr` must point to n elements
/// - `out_ptr` must have space for n elements
/// - `pivots_ptr` must point to k pivot indices (k = min(m, n) from original LU)
pub unsafe fn launch_apply_lu_permutation(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    in_ptr: u64,
    out_ptr: u64,
    pivots_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "apply_lu_permutation_f32",
        DType::F64 => "apply_lu_permutation_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "apply_lu_permutation",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&in_ptr);
    builder.arg(&out_ptr);
    builder.arg(&pivots_ptr);
    builder.arg(&n_u32);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA apply_lu_permutation kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

// ============================================================================
// Matrix Copy
// ============================================================================

/// Copy matrix data on device.
///
/// # Safety
///
/// - `src_ptr` and `dst_ptr` must point to n elements
#[allow(dead_code)]
pub unsafe fn launch_matrix_copy(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    src_ptr: u64,
    dst_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "matrix_copy_f32",
        DType::F64 => "matrix_copy_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "matrix_copy",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(n);
    let block = (BLOCK_SIZE, 1, 1);
    let n_u32 = n as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&src_ptr);
    builder.arg(&dst_ptr);
    builder.arg(&n_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA matrix_copy kernel launch failed: {:?}", e)))?;

    Ok(())
}

// ============================================================================
// Scatter Column - write vector to matrix column
// ============================================================================

/// Scatter vector into a column of a matrix (for GPU-only inverse).
///
/// # Safety
///
/// - `vec_ptr` must point to n elements
/// - `matrix_ptr` must point to n×n matrix
/// - `col` must be < n
pub unsafe fn launch_scatter_column(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    vec_ptr: u64,
    matrix_ptr: u64,
    n: usize,
    col: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "scatter_column_f32",
        DType::F64 => "scatter_column_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "scatter_column",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(n);
    let block = (BLOCK_SIZE, 1, 1);
    let n_u32 = n as u32;
    let col_u32 = col as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&vec_ptr);
    builder.arg(&matrix_ptr);
    builder.arg(&n_u32);
    builder.arg(&col_u32);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!("CUDA scatter_column kernel launch failed: {:?}", e))
    })?;

    Ok(())
}

// ============================================================================
// Count Above Threshold - for matrix_rank
// ============================================================================

/// Count elements with absolute value above threshold.
///
/// # Safety
///
/// - `values_ptr` must point to n elements
/// - `count_ptr` must point to a single u32 (zero-initialized)
pub unsafe fn launch_count_above_threshold(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    values_ptr: u64,
    count_ptr: u64,
    n: usize,
    threshold: f64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "count_above_threshold_f32",
        DType::F64 => "count_above_threshold_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "count_above_threshold",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(n);
    let block = (BLOCK_SIZE, 1, 1);
    let n_u32 = n as u32;

    // Shared memory for reduction (u32 count per thread)
    let shared_mem = BLOCK_SIZE * std::mem::size_of::<u32>() as u32;

    // Pre-compute threshold in both formats to ensure proper lifetime
    let thresh_f32 = threshold as f32;

    let cfg = launch_config(grid, block, shared_mem);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&values_ptr);
    builder.arg(&count_ptr);
    builder.arg(&n_u32);

    match dtype {
        DType::F32 => {
            builder.arg(&thresh_f32);
        }
        DType::F64 => {
            builder.arg(&threshold);
        }
        _ => unreachable!(),
    }

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA count_above_threshold kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

// ============================================================================
// Max Absolute Value
// ============================================================================

/// Compute maximum absolute value of elements.
///
/// # Safety
///
/// - `values_ptr` must point to n elements
/// - `max_ptr` must point to a single element (zero-initialized)
pub unsafe fn launch_max_abs(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    values_ptr: u64,
    max_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "max_abs_f32",
        DType::F64 => "max_abs_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "max_abs",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(n);
    let block = (BLOCK_SIZE, 1, 1);
    let n_u32 = n as u32;

    // Shared memory for reduction
    let shared_mem = BLOCK_SIZE * std::mem::size_of::<f64>() as u32;

    let cfg = launch_config(grid, block, shared_mem);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&values_ptr);
    builder.arg(&max_ptr);
    builder.arg(&n_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA max_abs kernel launch failed: {:?}", e)))?;

    Ok(())
}

// ============================================================================
// Create Identity Matrix
// ============================================================================

/// Create identity matrix on GPU.
///
/// # Safety
///
/// - `out_ptr` must have space for n×n elements
pub unsafe fn launch_create_identity(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    out_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "create_identity_f32",
        DType::F64 => "create_identity_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "create_identity",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let total = n * n;
    let grid = elementwise_launch_config(total);
    let block = (BLOCK_SIZE, 1, 1);
    let n_u32 = n as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&out_ptr);
    builder.arg(&n_u32);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA create_identity kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

// ============================================================================
// Extract Column
// ============================================================================

/// Extract a column from a matrix.
///
/// # Safety
///
/// - `matrix_ptr` must point to m×n_cols matrix
/// - `col_out_ptr` must have space for m elements
/// - `col` must be < n_cols
pub unsafe fn launch_extract_column(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    matrix_ptr: u64,
    col_out_ptr: u64,
    m: usize,
    n_cols: usize,
    col: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "extract_column_f32",
        DType::F64 => "extract_column_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "extract_column",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(m);
    let block = (BLOCK_SIZE, 1, 1);
    let m_u32 = m as u32;
    let n_cols_u32 = n_cols as u32;
    let col_u32 = col as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&matrix_ptr);
    builder.arg(&col_out_ptr);
    builder.arg(&m_u32);
    builder.arg(&n_cols_u32);
    builder.arg(&col_u32);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!("CUDA extract_column kernel launch failed: {:?}", e))
    })?;

    Ok(())
}

// ============================================================================
// SVD Jacobi Decomposition
// ============================================================================

/// Launch SVD Jacobi decomposition kernel.
///
/// This implements the One-Sided Jacobi algorithm for SVD.
/// After this kernel:
/// - `b_ptr` contains the normalized U matrix columns
/// - `v_ptr` contains V (to be transposed for V^T)
/// - `s_ptr` contains the singular values (unsorted)
///
/// # Safety
///
/// - `b_ptr` must point to [work_m, work_n] matrix (modified in-place to become U)
/// - `v_ptr` must have space for [work_n, work_n] matrix (V)
/// - `s_ptr` must have space for work_n elements (singular values)
/// - `converged_flag_ptr` must point to a single i32 (zero-initialized)
pub unsafe fn launch_svd_jacobi(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    b_ptr: u64,
    v_ptr: u64,
    s_ptr: u64,
    converged_flag_ptr: u64,
    work_m: usize,
    work_n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "svd_jacobi_f32",
        DType::F64 => "svd_jacobi_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "svd_jacobi",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    // Single thread kernel for sequential algorithm (ensures backend parity)
    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let work_m_u32 = work_m as u32;
    let work_n_u32 = work_n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&b_ptr);
    builder.arg(&v_ptr);
    builder.arg(&s_ptr);
    builder.arg(&work_m_u32);
    builder.arg(&work_n_u32);
    builder.arg(&converged_flag_ptr);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA svd_jacobi kernel launch failed: {:?}", e)))?;

    Ok(())
}

// ============================================================================
// Matrix Transpose - Optimized with shared memory tiling
// ============================================================================

/// Launch optimized matrix transpose kernel using shared memory tiling.
///
/// Transposes matrix in [rows, cols] -> out [cols, rows] using 32x32 tiles
/// with shared memory to achieve coalesced memory access patterns.
///
/// # Safety
///
/// - `input_ptr` must point to a valid [rows, cols] matrix
/// - `output_ptr` must point to allocated [cols, rows] matrix
pub unsafe fn launch_transpose(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "transpose_f32",
        DType::F64 => "transpose_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "transpose",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    // Tile dimensions must match TILE_DIM and BLOCK_ROWS in CUDA kernel
    const TILE_DIM: usize = 32;
    const BLOCK_ROWS: usize = 8;

    // Grid covers the matrix with tiles
    let grid_x = (cols + TILE_DIM - 1) / TILE_DIM;
    let grid_y = (rows + TILE_DIM - 1) / TILE_DIM;
    let cfg = launch_config(
        (grid_x as u32, grid_y as u32, 1),
        (TILE_DIM as u32, BLOCK_ROWS as u32, 1),
        0,
    );

    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&rows_u32);
    builder.arg(&cols_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA transpose kernel launch failed: {:?}", e)))?;

    Ok(())
}

// ============================================================================
// Eigendecomposition for Symmetric Matrices (Jacobi Algorithm)
// ============================================================================

/// Launch eigendecomposition kernel for symmetric matrices.
///
/// This implements the Jacobi eigenvalue algorithm for symmetric matrices.
/// After this kernel:
/// - `eigenvalues_ptr` contains the eigenvalues (unsorted)
/// - `eigenvectors_ptr` contains the eigenvector matrix V
///
/// # Safety
///
/// - `work_ptr` must point to [n, n] matrix (working copy, will be modified)
/// - `eigenvectors_ptr` must have space for [n, n] matrix
/// - `eigenvalues_ptr` must have space for n elements
/// - `converged_flag_ptr` must point to a single i32 (zero-initialized)
pub unsafe fn launch_eig_jacobi_symmetric(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    work_ptr: u64,
    eigenvectors_ptr: u64,
    eigenvalues_ptr: u64,
    converged_flag_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "eig_jacobi_symmetric_f32",
        DType::F64 => "eig_jacobi_symmetric_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "eig_jacobi_symmetric",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    // Single thread kernel for sequential algorithm (ensures backend parity)
    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work_ptr);
    builder.arg(&eigenvectors_ptr);
    builder.arg(&eigenvalues_ptr);
    builder.arg(&n_u32);
    builder.arg(&converged_flag_ptr);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA eig_jacobi_symmetric kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

// ============================================================================
// Schur Decomposition
// ============================================================================

/// Launch Schur decomposition kernel for general matrices.
///
/// Computes A = Z @ T @ Z^T where T is quasi-upper-triangular (real Schur form)
/// and Z is orthogonal.
///
/// # Safety
///
/// - `t_ptr` must point to [n, n] matrix (modified in-place to become T)
/// - `z_ptr` must have space for [n, n] matrix (orthogonal Z)
/// - `converged_flag_ptr` must point to a single i32 (zero-initialized)
pub unsafe fn launch_schur_decompose(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    t_ptr: u64,
    z_ptr: u64,
    converged_flag_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "schur_decompose_f32",
        DType::F64 => "schur_decompose_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "schur_decompose",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    // Single thread kernel for sequential algorithm
    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&t_ptr);
    builder.arg(&z_ptr);
    builder.arg(&n_u32);
    builder.arg(&converged_flag_ptr);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA schur_decompose kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

// ============================================================================
// General Eigenvalue Decomposition
// ============================================================================

/// Launch general eigenvalue decomposition kernel for non-symmetric matrices.
///
/// Computes eigenvalues (possibly complex) and eigenvectors for general matrices.
/// Uses Schur decomposition + back-substitution.
///
/// # Safety
///
/// - `t_ptr` must point to [n, n] matrix (working buffer, modified in-place)
/// - `z_ptr` must have space for [n, n] matrix (Schur vectors)
/// - `eval_real_ptr` must have space for n elements (real part of eigenvalues)
/// - `eval_imag_ptr` must have space for n elements (imaginary part of eigenvalues)
/// - `evec_real_ptr` must have space for [n, n] matrix (real part of eigenvectors)
/// - `evec_imag_ptr` must have space for [n, n] matrix (imaginary part of eigenvectors)
/// - `converged_flag_ptr` must point to a single i32 (zero-initialized)
pub unsafe fn launch_eig_general(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    t_ptr: u64,
    z_ptr: u64,
    eval_real_ptr: u64,
    eval_imag_ptr: u64,
    evec_real_ptr: u64,
    evec_imag_ptr: u64,
    converged_flag_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "eig_general_f32",
        DType::F64 => "eig_general_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "eig_general",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    // Single thread kernel for sequential algorithm
    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&t_ptr);
    builder.arg(&z_ptr);
    builder.arg(&eval_real_ptr);
    builder.arg(&eval_imag_ptr);
    builder.arg(&evec_real_ptr);
    builder.arg(&evec_imag_ptr);
    builder.arg(&n_u32);
    builder.arg(&converged_flag_ptr);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA eig_general kernel launch failed: {:?}", e)))?;

    Ok(())
}
