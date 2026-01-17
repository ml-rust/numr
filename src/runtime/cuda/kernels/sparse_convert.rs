//! Sparse format conversion CUDA kernel launchers
//!
//! This module provides Rust wrappers for GPU-native sparse format conversion kernels.

use cudarc::driver::safe::{CudaContext, CudaStream};
use cudarc::types::CudaTypeName;
use std::sync::Arc;

use super::loader::{get_kernel_function, get_or_load_module, launch_config};
use crate::error::{Error, Result};

// ============================================================================
// Pointer Expansion (CSR/CSC → COO)
// ============================================================================

/// Launch CSR/CSC → COO pointer expansion kernel
///
/// Expands compressed row/column pointers to explicit indices.
/// For CSR: row_ptrs[i] to row_ptrs[i+1] contains indices for row i
/// Output: row_indices[j] = i for all j in [row_ptrs[i], row_ptrs[i+1])
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers are valid device pointers
/// - ptrs has length n_major + 1
/// - indices_out has length nnz
pub unsafe fn launch_expand_ptrs(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    ptrs: u64,
    indices_out: u64,
    n_major: usize,
) -> Result<()> {
    let kernel_name = "expand_ptrs_i64";

    let module = get_or_load_module(context, device_index, "sparse_convert")?;
    let function = get_kernel_function(&module, kernel_name)?;

    let (grid, block) = launch_config(n_major);

    stream
        .launch_kernel(
            function,
            grid,
            block,
            0,
            &[
                ptrs.as_kernel_param(),
                indices_out.as_kernel_param(),
                (n_major as u32).as_kernel_param(),
            ],
        )
        .map_err(|e| Error::CudaKernelLaunchError {
            kernel: kernel_name.to_string(),
            source: e,
        })?;

    Ok(())
}

/// Launch CSC → CSR transpose kernel
///
/// Direct transpose without sorting. Uses scatter to write entries
/// to their correct positions in the transposed format.
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers are valid device pointers
/// - CSC arrays properly sized
/// - CSR output arrays properly allocated
pub unsafe fn launch_csc_to_csr_transpose<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    csc_col_ptrs: u64,
    csc_row_indices: u64,
    csc_values: u64,
    csr_row_ptrs: u64,
    csr_col_indices: u64,
    csr_values: u64,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "csc_to_csr_transpose_f32",
        "f64" => "csc_to_csr_transpose_f64",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for CSC→CSR transpose: {}",
                T::NAME
            )));
        }
    };

    let module = get_or_load_module(context, device_index, "sparse_convert")?;
    let function = get_kernel_function(&module, kernel_name)?;

    let (grid, block) = launch_config(ncols);

    stream
        .launch_kernel(
            function,
            grid,
            block,
            0,
            &[
                csc_col_ptrs.as_kernel_param(),
                csc_row_indices.as_kernel_param(),
                csc_values.as_kernel_param(),
                csr_row_ptrs.as_kernel_param(),
                csr_col_indices.as_kernel_param(),
                csr_values.as_kernel_param(),
                (nrows as u32).as_kernel_param(),
                (ncols as u32).as_kernel_param(),
            ],
        )
        .map_err(|e| Error::CudaKernelLaunchError {
            kernel: kernel_name.to_string(),
            source: e,
        })?;

    Ok(())
}

// ============================================================================
// Scalar Operations
// ============================================================================

/// Launch sparse scale kernel: values_out = scalar * values_in
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers are valid device pointers
/// - Both values arrays have length nnz
pub unsafe fn launch_sparse_scale<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    values_in: u64,
    values_out: u64,
    scalar: f64,
    nnz: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "sparse_scale_f32",
        "f64" => "sparse_scale_f64",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for sparse_scale: {}",
                T::NAME
            )));
        }
    };

    let module = get_or_load_module(context, device_index, "sparse_convert")?;
    let function = get_kernel_function(&module, kernel_name)?;

    let (grid, block) = launch_config(nnz);

    let scalar_typed = if T::NAME == "f32" {
        scalar as f32
    } else {
        scalar
    };

    stream
        .launch_kernel(
            function,
            grid,
            block,
            0,
            &[
                values_in.as_kernel_param(),
                values_out.as_kernel_param(),
                scalar_typed.as_kernel_param(),
                (nnz as u32).as_kernel_param(),
            ],
        )
        .map_err(|e| Error::CudaKernelLaunchError {
            kernel: kernel_name.to_string(),
            source: e,
        })?;

    Ok(())
}

// ============================================================================
// Reduction Operations
// ============================================================================

/// Launch sparse sum kernel: result = sum(values)
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers are valid device pointers
/// - values has length nnz
/// - result points to single element
pub unsafe fn launch_sparse_sum<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    values: u64,
    result: u64,
    nnz: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "sparse_sum_f32",
        "f64" => "sparse_sum_f64",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for sparse_sum: {}",
                T::NAME
            )));
        }
    };

    let module = get_or_load_module(context, device_index, "sparse_convert")?;
    let function = get_kernel_function(&module, kernel_name)?;

    let (grid, block) = launch_config(nnz);

    stream
        .launch_kernel(
            function,
            grid,
            block,
            0,
            &[
                values.as_kernel_param(),
                result.as_kernel_param(),
                (nnz as u32).as_kernel_param(),
            ],
        )
        .map_err(|e| Error::CudaKernelLaunchError {
            kernel: kernel_name.to_string(),
            source: e,
        })?;

    Ok(())
}

/// Launch sparse sum rows kernel (CSR format)
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers are valid device pointers
/// - row_ptrs has length nrows + 1
/// - values has length nnz
/// - row_sums has length nrows
pub unsafe fn launch_sparse_sum_rows_csr<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs: u64,
    values: u64,
    row_sums: u64,
    nrows: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "sparse_sum_rows_csr_f32",
        "f64" => "sparse_sum_rows_csr_f64",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for sparse_sum_rows: {}",
                T::NAME
            )));
        }
    };

    let module = get_or_load_module(context, device_index, "sparse_convert")?;
    let function = get_kernel_function(&module, kernel_name)?;

    let (grid, block) = launch_config(nrows);

    stream
        .launch_kernel(
            function,
            grid,
            block,
            0,
            &[
                row_ptrs.as_kernel_param(),
                values.as_kernel_param(),
                row_sums.as_kernel_param(),
                (nrows as u32).as_kernel_param(),
            ],
        )
        .map_err(|e| Error::CudaKernelLaunchError {
            kernel: kernel_name.to_string(),
            source: e,
        })?;

    Ok(())
}

/// Launch sparse nnz per row kernel (CSR format)
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers are valid device pointers
/// - row_ptrs has length nrows + 1
/// - row_nnz has length nrows
pub unsafe fn launch_sparse_nnz_per_row_csr(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs: u64,
    row_nnz: u64,
    nrows: usize,
) -> Result<()> {
    let kernel_name = "sparse_nnz_per_row_csr_i64";

    let module = get_or_load_module(context, device_index, "sparse_convert")?;
    let function = get_kernel_function(&module, kernel_name)?;

    let (grid, block) = launch_config(nrows);

    stream
        .launch_kernel(
            function,
            grid,
            block,
            0,
            &[
                row_ptrs.as_kernel_param(),
                row_nnz.as_kernel_param(),
                (nrows as u32).as_kernel_param(),
            ],
        )
        .map_err(|e| Error::CudaKernelLaunchError {
            kernel: kernel_name.to_string(),
            source: e,
        })?;

    Ok(())
}

// ============================================================================
// Histogram Operations
// ============================================================================

/// Launch histogram kernel to count NNZ per column from CSR format
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers are valid device pointers
/// - row_ptrs has length nrows + 1
/// - col_indices matches CSR format
/// - col_counts is zero-initialized and has length ncols
pub unsafe fn launch_histogram_csr_columns(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs: u64,
    col_indices: u64,
    col_counts: u64,
    nrows: usize,
) -> Result<()> {
    let kernel_name = "histogram_csr_columns_i64";

    let module = get_or_load_module(context, device_index, "sparse_convert")?;
    let function = get_kernel_function(&module, kernel_name)?;

    let (grid, block) = launch_config(nrows);

    stream
        .launch_kernel(
            function,
            grid,
            block,
            0,
            &[
                row_ptrs.as_kernel_param(),
                col_indices.as_kernel_param(),
                col_counts.as_kernel_param(),
                (nrows as u32).as_kernel_param(),
            ],
        )
        .map_err(|e| Error::CudaKernelLaunchError {
            kernel: kernel_name.to_string(),
            source: e,
        })?;

    Ok(())
}

/// Launch histogram kernel to count NNZ per row from CSC format
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers are valid device pointers
/// - col_ptrs has length ncols + 1
/// - row_indices matches CSC format
/// - row_counts is zero-initialized and has length nrows
pub unsafe fn launch_histogram_csc_rows(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    col_ptrs: u64,
    row_indices: u64,
    row_counts: u64,
    ncols: usize,
) -> Result<()> {
    let kernel_name = "histogram_csc_rows_i64";

    let module = get_or_load_module(context, device_index, "sparse_convert")?;
    let function = get_kernel_function(&module, kernel_name)?;

    let (grid, block) = launch_config(ncols);

    stream
        .launch_kernel(
            function,
            grid,
            block,
            0,
            &[
                col_ptrs.as_kernel_param(),
                row_indices.as_kernel_param(),
                row_counts.as_kernel_param(),
                (ncols as u32).as_kernel_param(),
            ],
        )
        .map_err(|e| Error::CudaKernelLaunchError {
            kernel: kernel_name.to_string(),
            source: e,
        })?;

    Ok(())
}

// ============================================================================
// CSR ↔ CSC Transpose
// ============================================================================

/// Launch CSR → CSC transpose kernel
///
/// Direct transpose without sorting. Uses scatter to write entries
/// to their correct positions in the transposed format.
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers are valid device pointers
/// - CSR arrays properly sized
/// - CSC output arrays properly allocated
pub unsafe fn launch_csr_to_csc_transpose<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    csr_row_ptrs: u64,
    csr_col_indices: u64,
    csr_values: u64,
    csc_col_ptrs: u64,
    csc_row_indices: u64,
    csc_values: u64,
    nrows: usize,
    ncols: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "csr_to_csc_transpose_f32",
        "f64" => "csr_to_csc_transpose_f64",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for CSR→CSC transpose: {}",
                T::NAME
            )));
        }
    };

    let module = get_or_load_module(context, device_index, "sparse_convert")?;
    let function = get_kernel_function(&module, kernel_name)?;

    let (grid, block) = launch_config(nrows);

    stream
        .launch_kernel(
            function,
            grid,
            block,
            0,
            &[
                csr_row_ptrs.as_kernel_param(),
                csr_col_indices.as_kernel_param(),
                csr_values.as_kernel_param(),
                csc_col_ptrs.as_kernel_param(),
                csc_row_indices.as_kernel_param(),
                csc_values.as_kernel_param(),
                (nrows as u32).as_kernel_param(),
                (ncols as u32).as_kernel_param(),
            ],
        )
        .map_err(|e| Error::CudaKernelLaunchError {
            kernel: kernel_name.to_string(),
            source: e,
        })?;

    Ok(())
}
