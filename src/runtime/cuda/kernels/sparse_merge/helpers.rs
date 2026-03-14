//! Helper utilities for sparse merge kernel launchers
//!
//! Shared infrastructure used by CSR and CSC merge operations:
//! - dtype suffix resolution
//! - generic count kernel launcher
//! - generic CSR/CSC compute kernel launchers
//! - exclusive scan wrapper

#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use cudarc::types::CudaTypeName;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::runtime::cuda::CudaRuntime;
use crate::tensor::Tensor;

use super::super::loader::{
    BLOCK_SIZE, get_kernel_function, get_or_load_module, kernel_names, launch_config,
};

// ============================================================================
// dtype suffix helper
// ============================================================================

/// Get dtype-specific kernel name suffix
pub(super) fn dtype_suffix<T: CudaTypeName>() -> Result<&'static str> {
    match T::NAME {
        "f32" => Ok("f32"),
        "f64" => Ok("f64"),
        "__half" => Ok("f16"),
        "__nv_bfloat16" => Ok("bf16"),
        _ => Err(Error::Internal(format!(
            "Unsupported dtype for sparse operation: {}",
            T::NAME
        ))),
    }
}

// ============================================================================
// Generic Kernel Launcher Helpers (DRY principle)
// ============================================================================

/// Generic launcher for kernels without dtype template (count kernels)
///
/// Eliminates duplication across count kernel launchers
///
/// # Safety
///
/// - `row_ptrs_a`, `col_indices_a`, `row_ptrs_b`, `col_indices_b`, and `row_counts` must be
///   valid device memory pointers on the device associated with `context`.
/// - `nrows` must match the number of rows in both sparse matrices.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub(super) unsafe fn launch_count_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    kernel_name: &str,
    row_ptrs_a: u64,
    col_indices_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    row_counts: u64,
    nrows: usize,
    error_context: &str,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
    let func = get_kernel_function(&module, kernel_name)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_i32 = nrows as i32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&row_ptrs_a);
    builder.arg(&col_indices_a);
    builder.arg(&row_ptrs_b);
    builder.arg(&col_indices_b);
    builder.arg(&row_counts);
    builder.arg(&nrows_i32);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("{} kernel launch failed: {:?}", error_context, e)))?;

    Ok(())
}

/// Generic launcher for dtype-templated compute kernels (CSR format)
///
/// Eliminates duplication across CSR add/sub/mul/div compute launchers
///
/// # Safety
///
/// - All pointer arguments (`row_ptrs_a`, `col_indices_a`, `values_a`, `row_ptrs_b`,
///   `col_indices_b`, `values_b`, `out_row_ptrs`, `out_col_indices`, `out_values`) must be
///   valid device memory pointers on the device associated with `context`.
/// - Output buffers must be pre-allocated to the correct sizes (determined by a prior count pass).
/// - `nrows` must match the number of rows in both input matrices.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub(super) unsafe fn launch_csr_compute_kernel<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    kernel_base_name: &str,
    row_ptrs_a: u64,
    col_indices_a: u64,
    values_a: u64,
    row_ptrs_b: u64,
    col_indices_b: u64,
    values_b: u64,
    out_row_ptrs: u64,
    out_col_indices: u64,
    out_values: u64,
    nrows: usize,
    error_context: &str,
) -> Result<()> {
    let suffix = dtype_suffix::<T>()?;
    let kernel_name = format!("{}_{}", kernel_base_name, suffix);

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_i32 = nrows as i32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&row_ptrs_a);
    builder.arg(&col_indices_a);
    builder.arg(&values_a);
    builder.arg(&row_ptrs_b);
    builder.arg(&col_indices_b);
    builder.arg(&values_b);
    builder.arg(&out_row_ptrs);
    builder.arg(&out_col_indices);
    builder.arg(&out_values);
    builder.arg(&nrows_i32);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("{} kernel launch failed: {:?}", error_context, e)))?;

    Ok(())
}

/// Generic launcher for dtype-templated compute kernels (CSC format)
///
/// Eliminates duplication across CSC add/sub/mul/div compute launchers
///
/// # Safety
///
/// - All pointer arguments (`col_ptrs_a`, `row_indices_a`, `values_a`, `col_ptrs_b`,
///   `row_indices_b`, `values_b`, `out_col_ptrs`, `out_row_indices`, `out_values`) must be
///   valid device memory pointers on the device associated with `context`.
/// - Output buffers must be pre-allocated to the correct sizes (determined by a prior count pass).
/// - `ncols` must match the number of columns in both input matrices.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub(super) unsafe fn launch_csc_compute_kernel<T: CudaTypeName>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    kernel_base_name: &str,
    col_ptrs_a: u64,
    row_indices_a: u64,
    values_a: u64,
    col_ptrs_b: u64,
    row_indices_b: u64,
    values_b: u64,
    out_col_ptrs: u64,
    out_row_indices: u64,
    out_values: u64,
    ncols: usize,
    error_context: &str,
) -> Result<()> {
    let suffix = dtype_suffix::<T>()?;
    let kernel_name = format!("{}_{}", kernel_base_name, suffix);

    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (ncols as u32 + block_size - 1) / block_size;
    let ncols_i32 = ncols as i32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&col_ptrs_a);
    builder.arg(&row_indices_a);
    builder.arg(&values_a);
    builder.arg(&col_ptrs_b);
    builder.arg(&row_indices_b);
    builder.arg(&values_b);
    builder.arg(&out_col_ptrs);
    builder.arg(&out_row_indices);
    builder.arg(&out_values);
    builder.arg(&ncols_i32);

    builder
        .launch(cfg)
        .map_err(|e| Error::Internal(format!("{} kernel launch failed: {:?}", error_context, e)))?;

    Ok(())
}

// ============================================================================
// Exclusive Scan (Prefix Sum)
// ============================================================================

/// Compute exclusive scan (prefix sum) on GPU tensor
///
/// Input: [3, 1, 4, 2]
/// Output: [0, 3, 4, 8, 10] (n+1 elements, last is total sum)
///
/// Uses GPU-native parallel scan (no CPU transfer)
pub(super) fn exclusive_scan_i32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input: &Tensor<CudaRuntime>,
) -> Result<(Tensor<CudaRuntime>, usize)> {
    let device = input.device();

    // Use GPU scan (imported from scan module)
    unsafe {
        super::super::scan::exclusive_scan_i32_gpu(context, stream, device_index, device, input)
    }
}
