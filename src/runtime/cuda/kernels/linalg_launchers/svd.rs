//! SVD kernel launcher: Jacobi algorithm

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{get_kernel_function, get_or_load_module, kernel_names, launch_config};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch SVD Jacobi decomposition kernel.
///
/// This implements the One-Sided Jacobi algorithm for SVD.
/// After this kernel:
/// - `b_ptr` contains the normalized U matrix columns
/// - `v_ptr` contains V (to be transposed for V^T)
/// - `s_ptr` contains the singular values (unsorted)
///
/// # Safety
///
/// - `b_ptr` must point to [work_m, work_n] matrix (modified in-place to become U)
/// - `v_ptr` must have space for [work_n, work_n] matrix (V)
/// - `s_ptr` must have space for work_n elements (singular values)
/// - `converged_flag_ptr` must point to a single i32 (zero-initialized)
pub unsafe fn launch_svd_jacobi(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    b_ptr: u64,
    v_ptr: u64,
    s_ptr: u64,
    converged_flag_ptr: u64,
    work_m: usize,
    work_n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_SVD_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "svd_jacobi_f32",
        DType::F64 => "svd_jacobi_f64",
        DType::F16 => "svd_jacobi_f16",
        DType::BF16 => "svd_jacobi_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "svd_jacobi",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;

    // Single thread kernel for sequential algorithm (ensures backend parity)
    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let work_m_u32 = work_m as u32;
    let work_n_u32 = work_n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&b_ptr);
    builder.arg(&v_ptr);
    builder.arg(&s_ptr);
    builder.arg(&work_m_u32);
    builder.arg(&work_n_u32);
    builder.arg(&converged_flag_ptr);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA svd_jacobi kernel launch failed: {:?}", e)))?;

    Ok(())
}
