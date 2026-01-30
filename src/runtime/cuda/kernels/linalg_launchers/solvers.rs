//! Solver kernel launchers: forward/backward substitution, determinant, permutation

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{get_kernel_function, get_or_load_module, kernel_names, launch_config};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch forward substitution kernel to solve Lx = b.
///
/// # Safety
///
/// - `l_ptr` must point to a valid [n, n] lower triangular matrix
/// - `b_ptr` must point to a valid vector of n elements
/// - `x_ptr` must have space for n elements
pub unsafe fn launch_forward_sub(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    l_ptr: u64,
    b_ptr: u64,
    x_ptr: u64,
    n: usize,
    unit_diagonal: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_SOLVERS_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "forward_sub_f32",
        DType::F64 => "forward_sub_f64",
        DType::F16 => "forward_sub_f16",
        DType::BF16 => "forward_sub_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "forward_sub",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;
    let unit_diag_i32 = if unit_diagonal { 1i32 } else { 0i32 };

    let mut builder = stream.launch_builder(&func);
    builder.arg(&l_ptr);
    builder.arg(&b_ptr);
    builder.arg(&x_ptr);
    builder.arg(&n_u32);
    builder.arg(&unit_diag_i32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA forward_sub kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Launch backward substitution kernel to solve Ux = b.
///
/// # Safety
///
/// - `u_ptr` must point to a valid [n, n] upper triangular matrix
/// - `b_ptr` must point to a valid vector of n elements
/// - `x_ptr` must have space for n elements
pub unsafe fn launch_backward_sub(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    u_ptr: u64,
    b_ptr: u64,
    x_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_SOLVERS_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "backward_sub_f32",
        DType::F64 => "backward_sub_f64",
        DType::F16 => "backward_sub_f16",
        DType::BF16 => "backward_sub_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "backward_sub",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&u_ptr);
    builder.arg(&b_ptr);
    builder.arg(&x_ptr);
    builder.arg(&n_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA backward_sub kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Launch determinant computation from LU decomposition.
///
/// # Safety
///
/// - `lu_ptr` must point to a valid [n, n] LU decomposition
/// - `det_ptr` must point to a single element
pub unsafe fn launch_det_from_lu(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    lu_ptr: u64,
    det_ptr: u64,
    n: usize,
    num_swaps: i32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_SOLVERS_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "det_from_lu_f32",
        DType::F64 => "det_from_lu_f64",
        DType::F16 => "det_from_lu_f16",
        DType::BF16 => "det_from_lu_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "det_from_lu",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&lu_ptr);
    builder.arg(&det_ptr);
    builder.arg(&n_u32);
    builder.arg(&num_swaps);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA det_from_lu kernel launch failed: {:?}", e)))?;

    Ok(())
}

/// Apply LU permutation to a vector.
///
/// # Safety
///
/// - `in_ptr` must point to n elements
/// - `out_ptr` must have space for n elements
/// - `pivots_ptr` must point to pivot indices
pub unsafe fn launch_apply_lu_permutation(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    in_ptr: u64,
    out_ptr: u64,
    pivots_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_SOLVERS_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "apply_lu_permutation_f32",
        DType::F64 => "apply_lu_permutation_f64",
        DType::F16 => "apply_lu_permutation_f16",
        DType::BF16 => "apply_lu_permutation_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "apply_lu_permutation",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&in_ptr);
    builder.arg(&out_ptr);
    builder.arg(&pivots_ptr);
    builder.arg(&n_u32);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA apply_lu_permutation kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}
