//! FP8 matmul CUDA kernel launchers
//!
//! Launches FP8 GEMM kernels with per-tensor scaling and F32 accumulation.
//! Output can be F32, F16, or BF16.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{get_kernel_function, get_or_load_module, launch_config};
use crate::dtype::DType;
use crate::error::{Error, Result};

const FP8_MATMUL_MODULE: &str = "fp8_matmul";

// Tile config matching the .cu defines
const TILE_M: u32 = 64;
const TILE_N: u32 = 64;
const THREAD_M: u32 = 4;
const THREAD_N: u32 = 4;

fn fp8_matmul_launch_cfg(m: usize, n: usize, batch: usize) -> super::loader::LaunchConfig {
    let grid_x = ((n as u32) + TILE_N - 1) / TILE_N;
    let grid_y = ((m as u32) + TILE_M - 1) / TILE_M;
    let threads_x = TILE_N / THREAD_N;
    let threads_y = TILE_M / THREAD_M;
    launch_config(
        (grid_x, grid_y, (batch as u32).max(1)),
        (threads_x, threads_y, 1),
        0,
    )
}

fn out_dtype_suffix(out_dtype: DType) -> Result<&'static str> {
    match out_dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        _ => Err(Error::UnsupportedDType {
            dtype: out_dtype,
            op: "fp8_matmul output",
        }),
    }
}

/// Launch FP8 E4M3 x E4M3 matmul kernel.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_fp8_matmul_e4m3(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    out_dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    scale_a: f32,
    scale_b: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FP8_MATMUL_MODULE)?;
    let suffix = out_dtype_suffix(out_dtype)?;
    let func_name = format!("fp8_matmul_e4m3_{}", suffix);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = fp8_matmul_launch_cfg(m, n, 1);
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&scale_a);
        builder.arg(&scale_b);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA fp8_matmul_e4m3 kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch FP8 E5M2 x E4M3 matmul kernel (backward pass).
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_fp8_matmul_e5m2(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    out_dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    scale_a: f32,
    scale_b: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FP8_MATMUL_MODULE)?;
    let suffix = out_dtype_suffix(out_dtype)?;
    let func_name = format!("fp8_matmul_e5m2_{}", suffix);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = fp8_matmul_launch_cfg(m, n, 1);
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&scale_a);
        builder.arg(&scale_b);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA fp8_matmul_e5m2 kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch batched FP8 E4M3 x E4M3 matmul kernel.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_fp8_matmul_e4m3_batched(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    out_dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    scale_a: f32,
    scale_b: f32,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FP8_MATMUL_MODULE)?;
    let suffix = out_dtype_suffix(out_dtype)?;
    let func_name = format!("fp8_matmul_e4m3_batched_{}", suffix);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = fp8_matmul_launch_cfg(m, n, batch);
    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&scale_a);
        builder.arg(&scale_b);
        builder.arg(&batch_u32);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA fp8_matmul_e4m3_batched kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch batched FP8 E5M2 x E4M3 matmul kernel (backward pass).
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_fp8_matmul_e5m2_batched(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    out_dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    scale_a: f32,
    scale_b: f32,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FP8_MATMUL_MODULE)?;
    let suffix = out_dtype_suffix(out_dtype)?;
    let func_name = format!("fp8_matmul_e5m2_batched_{}", suffix);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = fp8_matmul_launch_cfg(m, n, batch);
    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&scale_a);
        builder.arg(&scale_b);
        builder.arg(&batch_u32);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA fp8_matmul_e5m2_batched kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}
