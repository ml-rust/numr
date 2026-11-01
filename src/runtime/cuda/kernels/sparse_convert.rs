//! Launchers for the sparse format conversion kernels in `sparse_convert.cu`.
//! Every index buffer holds I64.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, LaunchConfig, get_kernel_function, get_or_load_module, kernel_names, launch_config,
};
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;

fn linear_config(n: usize) -> LaunchConfig {
    let grid_size = (n as u32).div_ceil(BLOCK_SIZE);
    launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0)
}

/// Expand `n_major + 1` compressed pointers into one index per entry:
/// `indices_out[j] = b` for every `j` in `ptrs[b]..ptrs[b + 1]`.
///
/// # Safety
///
/// `ptrs` must hold `n_major + 1` I64 elements and `indices_out` `ptrs[n_major]`
/// I64 elements, both live device buffers on `context`.
pub unsafe fn launch_expand_ptrs(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    ptrs: u64,
    indices_out: u64,
    n_major: usize,
) -> Result<()> {
    let kernel = "expand_ptrs_i64";
    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_CONVERT_MODULE)?;
    let function = get_kernel_function(&module, kernel)?;
    let n_u32 = n_major as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&ptrs);
    builder.arg(&indices_out);
    builder.arg(&n_u32);
    unsafe { builder.launch(linear_config(n_major)) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA {kernel} launch over {n_major} pointers failed: {e:?}"
        ))
    })?;
    Ok(())
}

/// Build `n_rows_or_cols + 1` compressed pointers from `nnz` sorted I64
/// indices: `ptrs_out[b]` = the number of indices below `b`.
///
/// # Safety
///
/// `sorted_indices` must hold `nnz` ascending I64 elements and `ptrs_out`
/// `n_rows_or_cols + 1` I64 elements, both live device buffers on `context`.
pub unsafe fn launch_build_ptrs_from_sorted(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    sorted_indices: u64,
    ptrs_out: u64,
    nnz: usize,
    n_rows_or_cols: usize,
) -> Result<()> {
    let kernel = "build_ptrs_from_sorted_indices_i64";
    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_CONVERT_MODULE)?;
    let function = get_kernel_function(&module, kernel)?;
    let nnz_u32 = nnz as u32;
    let n_u32 = n_rows_or_cols as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&sorted_indices);
    builder.arg(&ptrs_out);
    builder.arg(&nnz_u32);
    builder.arg(&n_u32);
    unsafe { builder.launch(linear_config(n_rows_or_cols + 1)) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA {kernel} launch (nnz={nnz}, n={n_rows_or_cols}) failed: {e:?}"
        ))
    })?;
    Ok(())
}
