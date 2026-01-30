//! Advanced decomposition kernel launchers: rsf2csf (from linalg_advanced.cu), QZ (from linalg_qz.cu)

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{get_kernel_function, get_or_load_module, kernel_names, launch_config};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch rsf2csf kernel to convert real Schur form to complex Schur form.
///
/// Processes 2x2 blocks on the diagonal representing complex conjugate eigenvalue
/// pairs and transforms them to upper triangular form in complex space.
///
/// # Safety
///
/// - `z_in_ptr` must point to [n, n] orthogonal matrix Z from Schur decomposition
/// - `t_in_ptr` must point to [n, n] quasi-upper-triangular matrix T
/// - `z_real_ptr`, `z_imag_ptr` must have space for [n, n] matrices (complex Z output)
/// - `t_real_ptr`, `t_imag_ptr` must have space for [n, n] matrices (complex T output)
pub unsafe fn launch_rsf2csf(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    z_in_ptr: u64,
    t_in_ptr: u64,
    z_real_ptr: u64,
    z_imag_ptr: u64,
    t_real_ptr: u64,
    t_imag_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_ADVANCED_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "rsf2csf_f32",
        DType::F64 => "rsf2csf_f64",
        DType::F16 => "rsf2csf_f16",
        DType::BF16 => "rsf2csf_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "rsf2csf",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&z_in_ptr);
    builder.arg(&t_in_ptr);
    builder.arg(&z_real_ptr);
    builder.arg(&z_imag_ptr);
    builder.arg(&t_real_ptr);
    builder.arg(&t_imag_ptr);
    builder.arg(&n_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA rsf2csf kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Launch QZ decomposition kernel for matrix pencil (A, B).
///
/// Computes the generalized Schur decomposition:
/// - A = Q @ S @ Z^T
/// - B = Q @ T @ Z^T
///
/// Uses Hessenberg-triangular reduction followed by Francis's implicit
/// double-shift QZ iteration (works entirely in real arithmetic).
///
/// # Safety
///
/// - `s_ptr` must point to [n, n] matrix (copy of A, modified in-place to become S)
/// - `t_ptr` must point to [n, n] matrix (copy of B, modified in-place to become T)
/// - `q_ptr` must have space for [n, n] matrix (left orthogonal factor Q)
/// - `z_ptr` must have space for [n, n] matrix (right orthogonal factor Z)
/// - `eig_real_ptr` must have space for n elements (real part of generalized eigenvalues)
/// - `eig_imag_ptr` must have space for n elements (imaginary part)
/// - `converged_flag_ptr` must point to a single i32 (zero-initialized)
pub unsafe fn launch_qz_decompose(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    s_ptr: u64,
    t_ptr: u64,
    q_ptr: u64,
    z_ptr: u64,
    eig_real_ptr: u64,
    eig_imag_ptr: u64,
    converged_flag_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_QZ_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "qz_decompose_f32",
        DType::F64 => "qz_decompose_f64",
        DType::F16 => "qz_decompose_f16",
        DType::BF16 => "qz_decompose_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "qz_decompose",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&s_ptr);
    builder.arg(&t_ptr);
    builder.arg(&q_ptr);
    builder.arg(&z_ptr);
    builder.arg(&eig_real_ptr);
    builder.arg(&eig_imag_ptr);
    builder.arg(&converged_flag_ptr);
    builder.arg(&n_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA qz_decompose kernel launch failed: {:?}", e)))?;

    Ok(())
}
