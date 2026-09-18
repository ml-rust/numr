//! GEMV launcher for the half-precision small-M decode path against a
//! transposed weight.
//!
//! One warp per pair of output columns, reducing along K lane-strided and
//! then through a shuffle tree. That float sequence differs from the tiled
//! GEMM's one-FMA-per-k order, so a row computed here is not the same bits as
//! the same row past the M cutoff; F32 no longer comes here for that reason
//! (`ops/helpers/matmul.rs`). F16 and BF16 still do, until their tensor-core
//! kernel stages a `[N, K]` weight directly.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};

use super::launch_dims::LaunchConfig;
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::{kernel_name, kernel_names};

/// Launch multi-row GEMV kernel with transposed B: C[batch,M,N] = A[batch,M,K] @ B^T
///
/// Each warp computes 2 output columns, sharing the activation vector load across rows.
/// Compiled for F16 and BF16 only; any other dtype is an error here, not a
/// missing-symbol failure at load.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
/// `b_ptr` points to the raw [N,K] data (NOT the transposed [K,N] view).
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gemv_kernel_bt_mr(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    if !matches!(dtype, DType::F16 | DType::BF16) {
        return Err(Error::Internal(format!(
            "gemv_bt_mr is compiled for F16 and BF16 only, got {dtype:?}; \
             the tiled matmul serves every other dtype at every M"
        )));
    }
    let module = get_or_load_module(context, device_index, kernel_names::GEMV_MODULE)?;
    let func_name = kernel_name("gemv_bt_mr", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    // grid: (ceil(N / (WARPS_PER_BLOCK * ROWS_PER_WARP)), M, batch), block: (256, 1, 1)
    // 8 warps per block, each warp handles 2 output columns.
    let warps_per_block: u32 = 8;
    let rows_per_warp: u32 = 2;
    let cols_per_block = warps_per_block * rows_per_warp; // 16
    let grid_x = (n as u32).div_ceil(cols_per_block);
    let grid_y = m as u32;
    let grid_z = batch as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let a_batch_u32 = a_batch as u32;
    let b_batch_u32 = b_batch as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA GEMV-BT-MR kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}
