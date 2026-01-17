//! Sparse matrix operations CUDA kernel launchers
//!
//! This module provides Rust wrappers for CUDA sparse matrix kernels.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use cudarc::types::CudaTypeName;
use std::sync::Arc;

use super::loader::{get_kernel_function, get_or_load_module, kernel_names, launch_config};
use crate::error::{Error, Result};

// ============================================================================
// SpMV Launchers (Row-per-thread)
// ============================================================================

/// Launch CSR SpMV kernel (row-per-thread variant)
///
/// y = A * x where A is sparse CSR matrix
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers are valid device pointers
/// - row_ptrs has length nrows + 1
/// - col_indices and values have length nnz
/// - x has length ncols
/// - y has length nrows
pub unsafe fn launch_csr_spmv<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs: u64,
    col_indices: u64,
    values: u64,
    x: u64,
    y: u64,
    nrows: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "csr_spmv_f32",
        "f64" => "csr_spmv_f64",
        "__half" => "csr_spmv_f16",
        "__nv_bfloat16" => "csr_spmv_bf16",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for sparse SpMV: {}",
                T::NAME
            )));
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SPARSE_SPMV_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let block_size = 256;
        let grid_size = (nrows + block_size - 1) / block_size;

        let cfg = launch_config((grid_size as u32, 1, 1), (block_size as u32, 1, 1), 0);
        let nrows_i32 = nrows as i32;
        let mut builder = stream.launch_builder(&func);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&values);
        builder.arg(&x);
        builder.arg(&y);
        builder.arg(&nrows_i32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA sparse SpMV kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

// ============================================================================
// SpMV Launchers (Warp-level reduction)
// ============================================================================

/// Launch CSR SpMV kernel (warp-level reduction variant)
///
/// Better for very sparse matrices where each row has few non-zeros.
///
/// # Safety
///
/// Same safety requirements as `launch_csr_spmv`
pub unsafe fn launch_csr_spmv_warp<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs: u64,
    col_indices: u64,
    values: u64,
    x: u64,
    y: u64,
    nrows: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "csr_spmv_warp_f32",
        "f64" => "csr_spmv_warp_f64",
        "__half" => "csr_spmv_warp_f16",
        "__nv_bfloat16" => "csr_spmv_warp_bf16",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for sparse SpMV warp: {}",
                T::NAME
            )));
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SPARSE_SPMV_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        // One block per row, 32 threads (one warp) per block
        let cfg = launch_config((nrows as u32, 1, 1), (32, 1, 1), 0);
        let nrows_i32 = nrows as i32;
        let mut builder = stream.launch_builder(&func);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&values);
        builder.arg(&x);
        builder.arg(&y);
        builder.arg(&nrows_i32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA sparse SpMV warp kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

// ============================================================================
// SpMM Launchers
// ============================================================================

/// Launch CSR SpMM kernel
///
/// C = A * B where A is sparse CSR matrix, B is dense matrix
///
/// # Safety
///
/// Caller must ensure:
/// - All pointers are valid device pointers
/// - row_ptrs has length nrows + 1
/// - col_indices and values have length nnz
/// - B has shape [ncols, ncols_B] stored row-major
/// - C has shape [nrows, ncols_B] stored row-major
pub unsafe fn launch_csr_spmm<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    row_ptrs: u64,
    col_indices: u64,
    values: u64,
    b: u64,
    c: u64,
    nrows: usize,
    ncols_b: usize,
) -> Result<()> {
    let kernel_name = match T::NAME {
        "f32" => "csr_spmm_f32",
        "f64" => "csr_spmm_f64",
        "__half" => "csr_spmm_f16",
        "__nv_bfloat16" => "csr_spmm_bf16",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for sparse SpMM: {}",
                T::NAME
            )));
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SPARSE_SPMV_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        // One block per row, ncols_b threads per block (up to 1024)
        let block_size = ncols_b.min(1024);
        let cfg = launch_config((nrows as u32, 1, 1), (block_size as u32, 1, 1), 0);
        let nrows_i32 = nrows as i32;
        let ncols_b_i32 = ncols_b as i32;
        let mut builder = stream.launch_builder(&func);
        builder.arg(&row_ptrs);
        builder.arg(&col_indices);
        builder.arg(&values);
        builder.arg(&b);
        builder.arg(&c);
        builder.arg(&nrows_i32);
        builder.arg(&ncols_b_i32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA sparse SpMM kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Choose optimal SpMV kernel based on matrix sparsity
///
/// Returns true if warp-level kernel should be used, false for row-per-thread
pub fn should_use_warp_kernel(avg_nnz_per_row: f32) -> bool {
    // Warp kernel is better when rows are very sparse (< 32 nnz per row)
    avg_nnz_per_row < 32.0
}
