//! Row and column sums, and stored entries per row or column.

#![allow(unsafe_op_in_unsafe_fn)]

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::super::loader::{BLOCK_SIZE, get_kernel_function, get_or_load_module, launch_config};
use super::common::{SPARSE_UTILS_MODULE, dtype_suffix};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::runtime::cuda::CudaRuntime;
use crate::runtime::cuda::GuardedStream;
use crate::tensor::Tensor;
// ============================================================================
// Row/Column Sums
// ============================================================================

/// Compute row-wise sum of a CSR sparse matrix (GPU kernel)
///
/// # Safety
///
/// - `row_ptrs` and `values` must be valid `CudaRuntime` tensors on the device associated with
///   `context`, with a consistent CSR structure where `row_ptrs` has `nrows + 1` elements.
/// - `nrows` must match the actual number of rows.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn csr_sum_rows_gpu<T: Element>(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_ptrs: &Tensor<CudaRuntime>,
    values: &Tensor<CudaRuntime>,
    nrows: usize,
) -> Result<Tensor<CudaRuntime>> {
    let suffix = dtype_suffix::<T>()?;
    let kernel_name = format!("csr_sum_rows_{}", suffix);

    let module = get_or_load_module(context, device_index, SPARSE_UTILS_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let out = Tensor::<CudaRuntime>::zeros(&[nrows], dtype, device)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_u32 = nrows as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);

    let row_ptrs_ptr = row_ptrs.ptr();
    let values_ptr = values.ptr();
    let out_ptr = out.ptr();

    let mut builder = stream.launch_builder(&func);
    builder.arg(&row_ptrs_ptr);
    builder.arg(&values_ptr);
    builder.arg(&out_ptr);
    builder.arg(&nrows_u32);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("CUDA csr_sum_rows kernel launch failed: {:?}", e)))?;

    Ok(out)
}

/// Compute column-wise sum of a CSC sparse matrix (GPU kernel)
///
/// # Safety
///
/// - `col_ptrs` and `values` must be valid `CudaRuntime` tensors on the device associated with
///   `context`, with a consistent CSC structure where `col_ptrs` has `ncols + 1` elements.
/// - `ncols` must match the actual number of columns.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn csc_sum_cols_gpu<T: Element>(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    col_ptrs: &Tensor<CudaRuntime>,
    values: &Tensor<CudaRuntime>,
    ncols: usize,
) -> Result<Tensor<CudaRuntime>> {
    let suffix = dtype_suffix::<T>()?;
    let kernel_name = format!("csc_sum_cols_{}", suffix);

    let module = get_or_load_module(context, device_index, SPARSE_UTILS_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let out = Tensor::<CudaRuntime>::zeros(&[ncols], dtype, device)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (ncols as u32 + block_size - 1) / block_size;
    let ncols_u32 = ncols as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);

    let col_ptrs_ptr = col_ptrs.ptr();
    let values_ptr = values.ptr();
    let out_ptr = out.ptr();

    let mut builder = stream.launch_builder(&func);
    builder.arg(&col_ptrs_ptr);
    builder.arg(&values_ptr);
    builder.arg(&out_ptr);
    builder.arg(&ncols_u32);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("CUDA csc_sum_cols kernel launch failed: {:?}", e)))?;

    Ok(out)
}

// ============================================================================
// NNZ Counting
// ============================================================================

/// Count non-zeros per row of a CSR matrix using pointer differences (GPU kernel)
///
/// # Safety
///
/// - `row_ptrs` must be a valid `CudaRuntime` tensor on the device associated with `context`,
///   with at least `nrows + 1` elements of type I64.
/// - `nrows` must match the actual number of rows.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn csr_nnz_per_row_gpu(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    row_ptrs: &Tensor<CudaRuntime>,
    nrows: usize,
) -> Result<Tensor<CudaRuntime>> {
    let module = get_or_load_module(context, device_index, SPARSE_UTILS_MODULE)?;
    let func = get_kernel_function(&module, "csr_nnz_per_row_kernel")?;

    let out = Tensor::<CudaRuntime>::zeros(&[nrows], DType::I64, device)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_u32 = nrows as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);

    let row_ptrs_ptr = row_ptrs.ptr();
    let out_ptr = out.ptr();

    let mut builder = stream.launch_builder(&func);
    builder.arg(&row_ptrs_ptr);
    builder.arg(&out_ptr);
    builder.arg(&nrows_u32);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA csr_nnz_per_row kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(out)
}

/// Count non-zeros per column of a CSC matrix using pointer differences (GPU kernel)
///
/// # Safety
///
/// - `col_ptrs` must be a valid `CudaRuntime` tensor on the device associated with `context`,
///   with at least `ncols + 1` elements of type I64.
/// - `ncols` must match the actual number of columns.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn csc_nnz_per_col_gpu(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    col_ptrs: &Tensor<CudaRuntime>,
    ncols: usize,
) -> Result<Tensor<CudaRuntime>> {
    let module = get_or_load_module(context, device_index, SPARSE_UTILS_MODULE)?;
    let func = get_kernel_function(&module, "csc_nnz_per_col_kernel")?;

    let out = Tensor::<CudaRuntime>::zeros(&[ncols], DType::I64, device)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (ncols as u32 + block_size - 1) / block_size;
    let ncols_u32 = ncols as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);

    let col_ptrs_ptr = col_ptrs.ptr();
    let out_ptr = out.ptr();

    let mut builder = stream.launch_builder(&func);
    builder.arg(&col_ptrs_ptr);
    builder.arg(&out_ptr);
    builder.arg(&ncols_u32);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA csc_nnz_per_col kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(out)
}
