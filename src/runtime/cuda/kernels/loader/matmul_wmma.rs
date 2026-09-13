//! Tensor-core WMMA GEMM launchers for F16 and BF16.
//!
//! `matmul_wmma_policy::use_wmma` decides when the path is legal; the
//! launchers below cover the 2-D and batched forms, each with and without a
//! fused bias. The bias + activation and bias + residual forms live in
//! `gemm_epilogue_wmma.rs` and share this module's launch geometry. Every
//! kernel family is instantiated at two block tiles; `matmul_wmma_tile` owns
//! the launch geometry and picks the tile per launch. The kernels use only
//! static shared memory, so the dynamic request is always zero.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};

use super::matmul_wmma_tile::{select_wmma_tile, wmma_kernel_name, wmma_launch_config};
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::kernel_names;

/// Launch 2-D (non-batched) WMMA GEMM for F16 or BF16.
///
/// # Safety
///
/// `a_ptr`, `b_ptr` and `c_ptr` must address `M×K`, `K×N` and `M×N` elements
/// of `dtype`. Any M, N, K >= 1 is accepted.
pub unsafe fn launch_matmul_wmma_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let tile = select_wmma_tile(m, n, k, 1, device_index);
    let func_name = wmma_kernel_name("matmul_wmma", dtype, tile);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = wmma_launch_config(m, n, 1, tile);

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA WMMA matmul kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch batched WMMA GEMM for F16 or BF16.
///
/// # Safety
///
/// `a_ptr`, `b_ptr` and `c_ptr` must address `a_batch×M×K`, `b_batch×K×N`
/// and `batch×M×N` elements of `dtype`. Any M, N, K >= 1 is accepted.
pub unsafe fn launch_matmul_wmma_batched_kernel(
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
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let tile = select_wmma_tile(m, n, k, batch, device_index);
    let func_name = wmma_kernel_name("matmul_wmma_batched", dtype, tile);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = wmma_launch_config(m, n, batch, tile);

    let batch_u32 = batch as u32;
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
        builder.arg(&batch_u32);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA WMMA batched matmul kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch 2-D (non-batched) WMMA GEMM with fused bias for F16 or BF16:
/// `C[M,N] = A[M,K] @ B[K,N] + bias[N]`.
///
/// The bias is added in F32, inside the epilogue, before the narrowing store.
///
/// # Safety
///
/// `a_ptr`, `b_ptr` and `c_ptr` must address `M×K`, `K×N` and `M×N` elements
/// of `dtype`, and `bias_ptr` N elements. Any M, N, K >= 1 is accepted.
pub unsafe fn launch_matmul_bias_wmma_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let tile = select_wmma_tile(m, n, k, 1, device_index);
    let func_name = wmma_kernel_name("matmul_bias_wmma", dtype, tile);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = wmma_launch_config(m, n, 1, tile);

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA WMMA matmul_bias kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch batched WMMA GEMM with fused bias for F16 or BF16. The bias is
/// `[N]` and broadcasts across rows and across batch slices.
///
/// # Safety
///
/// `a_ptr`, `b_ptr` and `c_ptr` must address `a_batch×M×K`, `b_batch×K×N`
/// and `batch×M×N` elements of `dtype`, and `bias_ptr` N elements. Any
/// M, N, K >= 1 is accepted.
pub unsafe fn launch_matmul_bias_wmma_batched_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let tile = select_wmma_tile(m, n, k, batch, device_index);
    let func_name = wmma_kernel_name("matmul_bias_wmma_batched", dtype, tile);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = wmma_launch_config(m, n, batch, tile);

    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let a_batch_u32 = a_batch as u32;
    let b_batch_u32 = b_batch as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&c_ptr);
        builder.arg(&batch_u32);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA WMMA batched matmul_bias kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}
