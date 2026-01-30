//! Eigendecomposition kernel launchers: symmetric, general, Schur

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{get_kernel_function, get_or_load_module, kernel_names, launch_config};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch eigendecomposition kernel for symmetric matrices.
///
/// This implements the Jacobi eigenvalue algorithm for symmetric matrices.
/// After this kernel:
/// - `eigenvalues_ptr` contains the eigenvalues (unsorted)
/// - `eigenvectors_ptr` contains the eigenvector matrix V
///
/// # Safety
///
/// - `work_ptr` must point to [n, n] matrix (working copy, will be modified)
/// - `eigenvectors_ptr` must have space for [n, n] matrix
/// - `eigenvalues_ptr` must have space for n elements
/// - `converged_flag_ptr` must point to a single i32 (zero-initialized)
pub unsafe fn launch_eig_jacobi_symmetric(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    work_ptr: u64,
    eigenvectors_ptr: u64,
    eigenvalues_ptr: u64,
    converged_flag_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_EIGEN_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "eig_jacobi_symmetric_f32",
        DType::F64 => "eig_jacobi_symmetric_f64",
        DType::F16 => "eig_jacobi_symmetric_f16",
        DType::BF16 => "eig_jacobi_symmetric_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "eig_jacobi_symmetric",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&work_ptr);
    builder.arg(&eigenvectors_ptr);
    builder.arg(&eigenvalues_ptr);
    builder.arg(&n_u32);
    builder.arg(&converged_flag_ptr);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA eig_jacobi_symmetric kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

/// Launch Schur decomposition kernel for general matrices.
///
/// Computes A = Z @ T @ Z^T where T is quasi-upper-triangular (real Schur form)
/// and Z is orthogonal.
///
/// # Safety
///
/// - `t_ptr` must point to [n, n] matrix (modified in-place to become T)
/// - `z_ptr` must have space for [n, n] matrix (orthogonal Z)
/// - `converged_flag_ptr` must point to a single i32 (zero-initialized)
pub unsafe fn launch_schur_decompose(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    t_ptr: u64,
    z_ptr: u64,
    converged_flag_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_SCHUR_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "schur_decompose_f32",
        DType::F64 => "schur_decompose_f64",
        DType::F16 => "schur_decompose_f16",
        DType::BF16 => "schur_decompose_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "schur_decompose",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&t_ptr);
    builder.arg(&z_ptr);
    builder.arg(&n_u32);
    builder.arg(&converged_flag_ptr);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA schur_decompose kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

/// Launch general eigenvalue decomposition kernel for non-symmetric matrices.
///
/// Computes eigenvalues (possibly complex) and eigenvectors for general matrices.
/// Uses Schur decomposition + back-substitution.
///
/// # Safety
///
/// - `t_ptr` must point to [n, n] matrix (working buffer, modified in-place)
/// - `z_ptr` must have space for [n, n] matrix (Schur vectors)
/// - `eval_real_ptr` must have space for n elements (real part of eigenvalues)
/// - `eval_imag_ptr` must have space for n elements (imaginary part of eigenvalues)
/// - `evec_real_ptr` must have space for [n, n] matrix (real part of eigenvectors)
/// - `evec_imag_ptr` must have space for [n, n] matrix (imaginary part of eigenvectors)
/// - `converged_flag_ptr` must point to a single i32 (zero-initialized)
pub unsafe fn launch_eig_general(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    t_ptr: u64,
    z_ptr: u64,
    eval_real_ptr: u64,
    eval_imag_ptr: u64,
    evec_real_ptr: u64,
    evec_imag_ptr: u64,
    converged_flag_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(
        context,
        device_index,
        kernel_names::LINALG_EIGEN_GENERAL_MODULE,
    )?;

    let func_name = match dtype {
        DType::F32 => "eig_general_f32",
        DType::F64 => "eig_general_f64",
        DType::F16 => "eig_general_f16",
        DType::BF16 => "eig_general_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "eig_general",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&t_ptr);
    builder.arg(&z_ptr);
    builder.arg(&eval_real_ptr);
    builder.arg(&eval_imag_ptr);
    builder.arg(&evec_real_ptr);
    builder.arg(&evec_imag_ptr);
    builder.arg(&n_u32);
    builder.arg(&converged_flag_ptr);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA eig_general kernel launch failed: {:?}", e)))?;

    Ok(())
}
