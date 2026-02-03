//! Matrix function kernel launchers for quasi-triangular matrices.
//!
//! Provides GPU kernels for computing matrix functions (exp, log, sqrt)
//! on Schur quasi-triangular matrices without GPU→CPU→GPU transfers.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, get_kernel_function, get_or_load_module, kernel_names, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch eigenvalue validation kernel for log or sqrt.
///
/// Checks if the quasi-triangular Schur matrix has problematic eigenvalues:
/// - For log: checks for non-positive real eigenvalues
/// - For sqrt: checks for negative real eigenvalues
///
/// # Safety
///
/// - `t_ptr` must point to a valid [n, n] quasi-triangular matrix
/// - `result_ptr` must point to 2 elements ([has_error, problematic_value])
pub unsafe fn launch_validate_eigenvalues(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    t_ptr: u64,
    result_ptr: u64,
    n: usize,
    eps: f64,
    mode: &str, // "log" or "sqrt"
) -> Result<()> {
    let module = get_or_load_module(
        context,
        device_index,
        kernel_names::LINALG_MATRIX_FUNCS_MODULE,
    )?;

    let func_name = match (dtype, mode) {
        (DType::F32, "log") => "validate_eigenvalues_log_f32",
        (DType::F64, "log") => "validate_eigenvalues_log_f64",
        (DType::F32, "sqrt") => "validate_eigenvalues_sqrt_f32",
        (DType::F64, "sqrt") => "validate_eigenvalues_sqrt_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "validate_eigenvalues",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;
    let eps_f32 = eps as f32;
    let eps_f64 = eps;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&t_ptr);
    builder.arg(&result_ptr);
    builder.arg(&n_u32);

    match dtype {
        DType::F32 => builder.arg(&eps_f32),
        DType::F64 => builder.arg(&eps_f64),
        _ => unreachable!(),
    };

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA validate_eigenvalues_{} kernel launch failed: {:?}",
            mode, e
        ))
    })?;

    Ok(())
}

/// Launch diagonal function kernel (exp, log, or sqrt on diagonal blocks).
///
/// Applies the function to 1x1 and 2x2 diagonal blocks of the quasi-triangular matrix.
///
/// # Safety
///
/// - `t_ptr` must point to a valid [n, n] quasi-triangular matrix
/// - `f_ptr` must point to allocated [n, n] output matrix (will be zeroed and filled)
pub unsafe fn launch_diagonal_func(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    t_ptr: u64,
    f_ptr: u64,
    n: usize,
    eps: f64,
    func_type: &str, // "exp", "log", or "sqrt"
) -> Result<()> {
    let module = get_or_load_module(
        context,
        device_index,
        kernel_names::LINALG_MATRIX_FUNCS_MODULE,
    )?;

    let func_name = match (dtype, func_type) {
        (DType::F32, "exp") => "diagonal_exp_f32",
        (DType::F64, "exp") => "diagonal_exp_f64",
        (DType::F32, "log") => "diagonal_log_f32",
        (DType::F64, "log") => "diagonal_log_f64",
        (DType::F32, "sqrt") => "diagonal_sqrt_f32",
        (DType::F64, "sqrt") => "diagonal_sqrt_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "diagonal_func",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;
    let eps_f32 = eps as f32;
    let eps_f64 = eps;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&t_ptr);
    builder.arg(&f_ptr);
    builder.arg(&n_u32);

    match dtype {
        DType::F32 => builder.arg(&eps_f32),
        DType::F64 => builder.arg(&eps_f64),
        _ => unreachable!(),
    };

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA diagonal_{} kernel launch failed: {:?}",
            func_type, e
        ))
    })?;

    Ok(())
}

/// Launch Parlett column computation kernel.
///
/// Computes off-diagonal elements F[i, col] for all i < col using Parlett's recurrence.
/// Must be called for columns 1..n in order (column-wise sequential algorithm).
///
/// # Safety
///
/// - `t_ptr` must point to a valid [n, n] quasi-triangular input matrix
/// - `f_ptr` must point to [n, n] output matrix with diagonal blocks already computed
pub unsafe fn launch_parlett_column(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    t_ptr: u64,
    f_ptr: u64,
    n: usize,
    col: usize,
    eps: f64,
) -> Result<()> {
    let module = get_or_load_module(
        context,
        device_index,
        kernel_names::LINALG_MATRIX_FUNCS_MODULE,
    )?;

    let func_name = match dtype {
        DType::F32 => "parlett_column_f32",
        DType::F64 => "parlett_column_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "parlett_column",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    // Each row i < col can be processed in parallel
    let grid_size = ((col as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let cfg = launch_config((grid_size.max(1), 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let n_u32 = n as u32;
    let col_u32 = col as u32;
    let eps_f32 = eps as f32;
    let eps_f64 = eps;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&t_ptr);
    builder.arg(&f_ptr);
    builder.arg(&n_u32);
    builder.arg(&col_u32);

    match dtype {
        DType::F32 => builder.arg(&eps_f32),
        DType::F64 => builder.arg(&eps_f64),
        _ => unreachable!(),
    };

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!("CUDA parlett_column kernel launch failed: {:?}", e))
    })?;

    Ok(())
}

/// Compute f(T) for quasi-triangular matrix T using GPU kernels.
///
/// This is the main entry point that:
/// 1. Applies function to diagonal blocks (GPU)
/// 2. Computes off-diagonal elements using Parlett's recurrence (GPU, column-by-column)
///
/// # Safety
///
/// - `t_ptr` must point to a valid [n, n] quasi-triangular input matrix
/// - `f_ptr` must point to allocated [n, n] output matrix
pub unsafe fn compute_schur_func_gpu(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    t_ptr: u64,
    f_ptr: u64,
    n: usize,
    func_type: &str,
) -> Result<()> {
    let eps = match dtype {
        DType::F32 => f32::EPSILON as f64,
        DType::F64 => f64::EPSILON,
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "compute_schur_func_gpu",
            });
        }
    };

    // Step 1: Apply function to diagonal blocks
    unsafe {
        launch_diagonal_func(
            context,
            stream,
            device_index,
            dtype,
            t_ptr,
            f_ptr,
            n,
            eps,
            func_type,
        )?;
    }

    // Step 2: Compute off-diagonal elements column by column
    // Parlett's algorithm requires column-wise processing
    for col in 1..n {
        unsafe {
            launch_parlett_column(
                context,
                stream,
                device_index,
                dtype,
                t_ptr,
                f_ptr,
                n,
                col,
                eps,
            )?;
        }
    }

    Ok(())
}
