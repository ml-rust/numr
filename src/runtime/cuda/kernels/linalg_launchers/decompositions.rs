//! Decomposition kernel launchers: LU, Cholesky, QR

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{get_kernel_function, get_or_load_module, kernel_names, launch_config};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch LU decomposition kernel with partial pivoting.
///
/// Modifies lu_ptr in-place to store L (below diagonal) and U (on/above diagonal).
///
/// # Safety
///
/// - `lu_ptr` must point to a valid [m, n] matrix (will be modified in-place)
/// - `pivots_ptr` must have space for min(m, n) i64 elements
/// - `num_swaps_ptr` must point to a single i32
/// - `singular_flag_ptr` must point to a single i32 (zero-initialized)
pub unsafe fn launch_lu_decompose(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    lu_ptr: u64,
    pivots_ptr: u64,
    num_swaps_ptr: u64,
    singular_flag_ptr: u64,
    m: usize,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_DECOMP_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "lu_decompose_f32",
        DType::F64 => "lu_decompose_f64",
        DType::F16 => "lu_decompose_f16",
        DType::BF16 => "lu_decompose_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "lu_decompose",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let m_u32 = m as u32;
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&lu_ptr);
    builder.arg(&pivots_ptr);
    builder.arg(&num_swaps_ptr);
    builder.arg(&m_u32);
    builder.arg(&n_u32);
    builder.arg(&singular_flag_ptr);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA lu_decompose kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Launch Cholesky decomposition kernel.
///
/// Modifies l_ptr in-place to store L (lower triangular factor).
///
/// # Safety
///
/// - `l_ptr` must point to a valid [n, n] symmetric positive-definite matrix
/// - `not_pd_flag_ptr` must point to a single i32 (zero-initialized)
pub unsafe fn launch_cholesky_decompose(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    l_ptr: u64,
    not_pd_flag_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_DECOMP_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "cholesky_decompose_f32",
        DType::F64 => "cholesky_decompose_f64",
        DType::F16 => "cholesky_decompose_f16",
        DType::BF16 => "cholesky_decompose_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "cholesky_decompose",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&l_ptr);
    builder.arg(&n_u32);
    builder.arg(&not_pd_flag_ptr);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA cholesky_decompose kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

/// Launch QR decomposition kernel using Householder reflections.
///
/// # Safety
///
/// - `q_ptr` must have space for [m, m] (full) or [m, k] (thin) matrix
/// - `r_ptr` must point to a copy of input matrix [m, n] (modified in-place)
/// - `workspace_ptr` must have space for m elements (Householder vector)
pub unsafe fn launch_qr_decompose(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    q_ptr: u64,
    r_ptr: u64,
    workspace_ptr: u64,
    m: usize,
    n: usize,
    thin: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_DECOMP_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "qr_decompose_f32",
        DType::F64 => "qr_decompose_f64",
        DType::F16 => "qr_decompose_f16",
        DType::BF16 => "qr_decompose_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "qr_decompose",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let thin_i32 = if thin { 1i32 } else { 0i32 };

    let mut builder = stream.launch_builder(&func);
    builder.arg(&q_ptr);
    builder.arg(&r_ptr);
    builder.arg(&workspace_ptr);
    builder.arg(&m_u32);
    builder.arg(&n_u32);
    builder.arg(&thin_i32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA qr_decompose kernel launch failed: {:?}", e)))?;

    Ok(())
}
