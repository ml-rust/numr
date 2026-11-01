//! Launchers for the SpGEMM kernels in `spgemm.cu`.
//!
//! C = A * B over CSR operands with I64 indices: a per-row product bound,
//! hash accumulation into global scratch, then a gather of the kept entries.
//! No row size or column count limit applies.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::loader::{BLOCK_SIZE, get_kernel_function, get_or_load_module, launch_config};
use super::sparse_dtype::float_kernel_name;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;

/// CUDA module name for sparse matrix-matrix multiplication (SpGEMM) kernels.
pub const SPGEMM_MODULE: &str = "spgemm";

/// Device pointers of one CSR operand of SpGEMM: I64 pointers and indices.
#[derive(Debug, Clone, Copy)]
pub struct SpgemmCsr {
    /// Row pointers, `m + 1` I64.
    pub row_ptrs: u64,
    /// Column indices, `nnz` I64.
    pub col_indices: u64,
    /// Values, `nnz` of the value dtype.
    pub values: u64,
}

fn launch_failed(kernel: &str, e: impl std::fmt::Debug) -> Error {
    Error::Internal(format!("CUDA {kernel} launch failed: {e:?}"))
}

/// `bound[i]` = the number of products A[i, k] * B[k, j] of output row `i`,
/// an upper bound of its entries. `bound` holds `m` I32 elements.
///
/// # Safety
///
/// Every pointer must be a live device buffer on `context` of the CSR length
/// it names, and `bound` must hold `m` I32 elements.
pub unsafe fn spgemm_row_bound(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    a: &SpgemmCsr,
    b: &SpgemmCsr,
    bound: u64,
    m: usize,
) -> Result<()> {
    let kernel = "spgemm_row_bound";
    let module = get_or_load_module(context, device_index, SPGEMM_MODULE)?;
    let func = get_kernel_function(&module, kernel)?;
    let grid = (m as u32).div_ceil(BLOCK_SIZE);
    let m_u32 = m as u32;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&a.row_ptrs);
    builder.arg(&a.col_indices);
    builder.arg(&b.row_ptrs);
    builder.arg(&bound);
    builder.arg(&m_u32);
    unsafe { builder.launch(launch_config((grid, 1, 1), (BLOCK_SIZE, 1, 1), 0)) }
        .map_err(|e| launch_failed(kernel, e))?;
    Ok(())
}

/// Accumulate every product of each output row into its scratch slice
/// `slot_ptrs[i]..slot_ptrs[i + 1]`, sort the slice by column and keep the
/// entries with `|value| > threshold`. `kept[i]` receives the kept count.
///
/// Scratch values are F64 for F64, F32 for every other dtype.
///
/// # Safety
///
/// Every pointer must be a live device buffer on `context`. `slot_cols` and
/// `slot_vals` must hold `slot_ptrs[m]` elements, and `kept` `m` I32 elements.
pub unsafe fn spgemm_accumulate(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    a: &SpgemmCsr,
    b: &SpgemmCsr,
    slot_ptrs: u64,
    slot_cols: u64,
    slot_vals: u64,
    kept: u64,
    m: usize,
    threshold: f64,
) -> Result<()> {
    let kernel = float_kernel_name("spgemm_accumulate", dtype, "sparse_matmul (CUDA)")?;
    let module = get_or_load_module(context, device_index, SPGEMM_MODULE)?;
    let func = get_kernel_function(&module, &kernel)?;
    let m_u32 = m as u32;
    let threshold_f32 = threshold as f32;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&a.row_ptrs);
    builder.arg(&a.col_indices);
    builder.arg(&a.values);
    builder.arg(&b.row_ptrs);
    builder.arg(&b.col_indices);
    builder.arg(&b.values);
    builder.arg(&slot_ptrs);
    builder.arg(&slot_cols);
    builder.arg(&slot_vals);
    builder.arg(&kept);
    builder.arg(&m_u32);
    if dtype == DType::F64 {
        builder.arg(&threshold);
    } else {
        builder.arg(&threshold_f32);
    }
    // One block per output row.
    unsafe { builder.launch(launch_config((m_u32, 1, 1), (BLOCK_SIZE, 1, 1), 0)) }
        .map_err(|e| launch_failed(&kernel, e))?;
    Ok(())
}

/// Copy the kept entries of each row from its scratch slice to the output
/// at `out_ptrs[i]`, converting scratch values to `dtype`.
///
/// # Safety
///
/// Every pointer must be a live device buffer on `context`. `out_cols` and
/// `out_vals` must hold `out_ptrs[m]` elements.
pub unsafe fn spgemm_gather(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    slot_ptrs: u64,
    slot_cols: u64,
    slot_vals: u64,
    out_ptrs: u64,
    out_cols: u64,
    out_vals: u64,
    m: usize,
) -> Result<()> {
    let kernel = float_kernel_name("spgemm_gather", dtype, "sparse_matmul (CUDA)")?;
    let module = get_or_load_module(context, device_index, SPGEMM_MODULE)?;
    let func = get_kernel_function(&module, &kernel)?;
    let grid = (m as u32).div_ceil(BLOCK_SIZE);
    let m_u32 = m as u32;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&slot_ptrs);
    builder.arg(&slot_cols);
    builder.arg(&slot_vals);
    builder.arg(&out_ptrs);
    builder.arg(&out_cols);
    builder.arg(&out_vals);
    builder.arg(&m_u32);
    unsafe { builder.launch(launch_config((grid, 1, 1), (BLOCK_SIZE, 1, 1), 0)) }
        .map_err(|e| launch_failed(&kernel, e))?;
    Ok(())
}
