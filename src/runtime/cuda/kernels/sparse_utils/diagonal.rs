//! CSR diagonal extraction.

#![allow(unsafe_op_in_unsafe_fn)]

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::super::loader::{BLOCK_SIZE, get_kernel_function, get_or_load_module, launch_config};
use super::common::{SPARSE_UTILS_MODULE, dtype_suffix};
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;
// ============================================================================
// CSR Diagonal Extraction
// ============================================================================

/// Extract diagonal elements from a CSR matrix (GPU kernel)
///
/// # Safety
///
/// - `row_ptrs`, `col_indices`, `values`, and `diag` must be valid device memory pointers on
///   the device associated with `context`.
/// - `row_ptrs` must have at least `n + 1` elements; `diag` must have at least `n` elements.
/// - `col_indices` and `values` must have at least `nnz` elements (as encoded in `row_ptrs`).
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn launch_csr_extract_diagonal<T: Element>(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    row_ptrs: u64,
    col_indices: u64,
    values: u64,
    diag: u64,
    n: usize,
) -> Result<()> {
    let suffix = dtype_suffix::<T>()?;
    let kernel_name = format!("csr_extract_diagonal_{}", suffix);

    let module = get_or_load_module(context, device_index, SPARSE_UTILS_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (n as u32 + block_size - 1) / block_size;
    let n_u32 = n as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&row_ptrs);
    builder.arg(&col_indices);
    builder.arg(&values);
    builder.arg(&diag);
    builder.arg(&n_u32);

    builder.launch(cfg).map_err(|e| {
        Error::Internal(format!(
            "CUDA csr_extract_diagonal kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}
