//! CUDA kernel launchers for sparse QR factorization
//!
//! Implements Householder QR reduction for sparse matrices on NVIDIA GPUs.
//! Five primitive kernels composed into a column-wise left-looking algorithm:
//!
//! - `apply_reflector`: Fused dot+axpy Householder update (single block, shared mem reduction)
//! - `norm`: Parallel sum-of-squares reduction for ||work[start..start+count]||^2
//! - `householder`: Householder vector generation with tau and R diagonal computation
//! - `extract_r`: Copy R off-diagonal entries from work vector
//! - `clear`: Zero-initialize work vector
//!
//! All single-block kernels use 256 threads with shared memory reductions.
//! Grid-based kernels (extract_r, clear) scale to arbitrary sizes.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::{
    BLOCK_SIZE, SPARSE_LINALG_MODULE, get_kernel_function, get_or_load_module, grid_size,
    launch_config, launch_error,
};
use crate::error::Result;

// ============================================================================
// Apply Householder Reflector (single block, fused dot + axpy)
// ============================================================================

/// Applies dense Householder reflector to work vector - f32
/// work[v_start..] -= tau * (v^T * work[v_start..]) * v
/// Single block of 256 threads with shared memory reduction.
pub unsafe fn launch_sparse_qr_apply_reflector_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    v: u64,
    v_start: i32,
    v_len: i32,
    tau_ptr: u64,
    work: u64,
    m: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_qr_apply_reflector_f32")?;
    let cfg = launch_config((1, 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&v);
    builder.arg(&v_start);
    builder.arg(&v_len);
    builder.arg(&tau_ptr);
    builder.arg(&work);
    builder.arg(&m);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_qr_apply_reflector_f32", e))?;
    Ok(())
}

/// Applies dense Householder reflector to work vector - f64
pub unsafe fn launch_sparse_qr_apply_reflector_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    v: u64,
    v_start: i32,
    v_len: i32,
    tau_ptr: u64,
    work: u64,
    m: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_qr_apply_reflector_f64")?;
    let cfg = launch_config((1, 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&v);
    builder.arg(&v_start);
    builder.arg(&v_len);
    builder.arg(&tau_ptr);
    builder.arg(&work);
    builder.arg(&m);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_qr_apply_reflector_f64", e))?;
    Ok(())
}

// ============================================================================
// Norm (sum of squares reduction, single block)
// ============================================================================

/// Computes ||work[start..start+count]||^2 via parallel reduction - f32
pub unsafe fn launch_sparse_qr_norm_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    work: u64,
    start: i32,
    count: i32,
    result: u64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_qr_norm_f32")?;
    let cfg = launch_config((1, 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work);
    builder.arg(&start);
    builder.arg(&count);
    builder.arg(&result);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_qr_norm_f32", e))?;
    Ok(())
}

/// Computes ||work[start..start+count]||^2 - f64
pub unsafe fn launch_sparse_qr_norm_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    work: u64,
    start: i32,
    count: i32,
    result: u64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_qr_norm_f64")?;
    let cfg = launch_config((1, 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work);
    builder.arg(&start);
    builder.arg(&count);
    builder.arg(&result);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_qr_norm_f64", e))?;
    Ok(())
}

// ============================================================================
// Householder vector computation (single block)
// ============================================================================

/// Computes Householder vector from work[start..m] - f32
pub unsafe fn launch_sparse_qr_householder_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    work: u64,
    start: i32,
    m: i32,
    norm_sq_ptr: u64,
    out_v: u64,
    out_tau: u64,
    out_diag: u64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_qr_householder_f32")?;
    let cfg = launch_config((1, 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work);
    builder.arg(&start);
    builder.arg(&m);
    builder.arg(&norm_sq_ptr);
    builder.arg(&out_v);
    builder.arg(&out_tau);
    builder.arg(&out_diag);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_qr_householder_f32", e))?;
    Ok(())
}

/// Computes Householder vector from work[start..m] - f64
pub unsafe fn launch_sparse_qr_householder_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    work: u64,
    start: i32,
    m: i32,
    norm_sq_ptr: u64,
    out_v: u64,
    out_tau: u64,
    out_diag: u64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_qr_householder_f64")?;
    let cfg = launch_config((1, 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work);
    builder.arg(&start);
    builder.arg(&m);
    builder.arg(&norm_sq_ptr);
    builder.arg(&out_v);
    builder.arg(&out_tau);
    builder.arg(&out_diag);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_qr_householder_f64", e))?;
    Ok(())
}

// ============================================================================
// Extract R off-diagonal entries
// ============================================================================

/// Copies work[0..count] to output buffer - f32
pub unsafe fn launch_sparse_qr_extract_r_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    work: u64,
    count: i32,
    output: u64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_qr_extract_r_f32")?;
    let cfg = launch_config((grid_size(count as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work);
    builder.arg(&count);
    builder.arg(&output);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_qr_extract_r_f32", e))?;
    Ok(())
}

/// Copies work[0..count] to output buffer - f64
pub unsafe fn launch_sparse_qr_extract_r_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    work: u64,
    count: i32,
    output: u64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_qr_extract_r_f64")?;
    let cfg = launch_config((grid_size(count as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work);
    builder.arg(&count);
    builder.arg(&output);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_qr_extract_r_f64", e))?;
    Ok(())
}

// ============================================================================
// Clear work vector
// ============================================================================

/// Sets work[0..n] to zero - f32
pub unsafe fn launch_sparse_qr_clear_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    work: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_qr_clear_f32")?;
    let cfg = launch_config((grid_size(n as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_qr_clear_f32", e))?;
    Ok(())
}

/// Sets work[0..n] to zero - f64
pub unsafe fn launch_sparse_qr_clear_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    work: u64,
    n: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SPARSE_LINALG_MODULE)?;
    let func = get_kernel_function(&module, "sparse_qr_clear_f64")?;
    let cfg = launch_config((grid_size(n as u32), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_error("sparse_qr_clear_f64", e))?;
    Ok(())
}
