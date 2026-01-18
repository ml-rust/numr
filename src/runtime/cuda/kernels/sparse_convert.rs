//! Sparse format conversion CUDA kernel launchers
//!
//! This module provides Rust wrappers for GPU-native sparse format conversion kernels.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use cudarc::types::CudaTypeName;
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, get_kernel_function, get_or_load_module, kernel_names, launch_config,
};
use crate::error::{Error, Result};

/// Helper to compute launch config from element count
fn compute_launch_config(n: usize) -> super::loader::LaunchConfig {
    let grid_size = (n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0)
}

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

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_CONVERT_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(n_major);

    let n_major_i32 = n_major as i32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&ptrs);
    builder.arg(&indices_out);
    builder.arg(&n_major_i32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

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

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_CONVERT_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(ncols);

    let nrows_i32 = nrows as i32;
    let ncols_i32 = ncols as i32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&csc_col_ptrs);
    builder.arg(&csc_row_indices);
    builder.arg(&csc_values);
    builder.arg(&csr_row_ptrs);
    builder.arg(&csr_col_indices);
    builder.arg(&csr_values);
    builder.arg(&nrows_i32);
    builder.arg(&ncols_i32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

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

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_CONVERT_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(nnz);

    let nnz_i32 = nnz as i32;
    let scalar_f32 = scalar as f32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&values_in);
    builder.arg(&values_out);
    if T::NAME == "f32" {
        builder.arg(&scalar_f32);
    } else {
        builder.arg(&scalar);
    }
    builder.arg(&nnz_i32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

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

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_CONVERT_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(nnz);

    let nnz_i32 = nnz as i32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&values);
    builder.arg(&result);
    builder.arg(&nnz_i32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

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

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_CONVERT_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(nrows);

    let nrows_i32 = nrows as i32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&row_ptrs);
    builder.arg(&values);
    builder.arg(&row_sums);
    builder.arg(&nrows_i32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

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

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_CONVERT_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(nrows);

    let nrows_i32 = nrows as i32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&row_ptrs);
    builder.arg(&row_nnz);
    builder.arg(&nrows_i32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

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

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_CONVERT_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(nrows);

    let nrows_i32 = nrows as i32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&row_ptrs);
    builder.arg(&col_indices);
    builder.arg(&col_counts);
    builder.arg(&nrows_i32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

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

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_CONVERT_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(ncols);

    let ncols_i32 = ncols as i32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&col_ptrs);
    builder.arg(&row_indices);
    builder.arg(&row_counts);
    builder.arg(&ncols_i32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

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

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_CONVERT_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(nrows);

    let nrows_i32 = nrows as i32;
    let ncols_i32 = ncols as i32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&csr_row_ptrs);
    builder.arg(&csr_col_indices);
    builder.arg(&csr_values);
    builder.arg(&csc_col_ptrs);
    builder.arg(&csc_row_indices);
    builder.arg(&csc_values);
    builder.arg(&nrows_i32);
    builder.arg(&ncols_i32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// COO → CSR/CSC: Build Pointers from Sorted Indices
// ============================================================================

/// Launch kernel to build row_ptrs or col_ptrs from sorted indices
///
/// After sorting COO entries, this builds the compressed pointer array.
/// For CSR: pass sorted row_indices to get row_ptrs
/// For CSC: pass sorted col_indices to get col_ptrs
///
/// # Safety
///
/// Caller must ensure:
/// - sorted_indices is sorted array of row/col indices (length nnz)
/// - ptrs_out is allocated with size n_rows_or_cols + 1
/// - All pointers are valid device pointers
pub unsafe fn launch_build_ptrs_from_sorted(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    sorted_indices: u64,
    ptrs_out: u64,
    nnz: usize,
    n_rows_or_cols: usize,
) -> Result<()> {
    let kernel_name = "build_ptrs_from_sorted_indices_i64";

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_CONVERT_MODULE)?;
    let function = get_kernel_function(&module, kernel_name)?;

    let cfg = compute_launch_config(nnz.max(1)); // At least 1 thread

    let nnz_u32 = nnz as u32;
    let n_rows_or_cols_u32 = n_rows_or_cols as u32;

    let mut builder = stream.launch_builder(&function);
    builder.arg(&sorted_indices);
    builder.arg(&ptrs_out);
    builder.arg(&nnz_u32);
    builder.arg(&n_rows_or_cols_u32);

    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed (nnz={}, n_rows_or_cols={}): {:?}",
                kernel_name, nnz, n_rows_or_cols, e
            ))
        })?;
    }

    Ok(())
}
