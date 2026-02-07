//! Banded solver kernel launchers

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{get_kernel_function, get_or_load_module, kernel_names, launch_config};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch banded solver kernel.
///
/// # Safety
///
/// - `ab_ptr` must point to a valid [band_rows * n] band storage matrix
///   where band_rows = kl + ku + 1, in LAPACK-style row-major format:
///   ab[(ku + i - j) * n + j] = A[i, j]
/// - `b_ptr` must point to a valid vector of n elements (RHS)
/// - `x_ptr` must have space for n elements (output solution)
/// - `work_ptr` must point to workspace of size [(2*kl+ku+1) * n] floats
///   (only used for general banded solver, not for Thomas algorithm)
pub unsafe fn launch_banded_solve(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    ab_ptr: u64,
    b_ptr: u64,
    x_ptr: u64,
    work_ptr: u64,
    n: usize,
    kl: usize,
    ku: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::LINALG_BANDED_MODULE)?;

    let func_name = match dtype {
        DType::F32 => "banded_solve_f32",
        DType::F64 => "banded_solve_f64",
        DType::F16 => "banded_solve_f16",
        DType::BF16 => "banded_solve_bf16",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "banded_solve",
            });
        }
    };

    let func = get_kernel_function(&module, func_name)?;
    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);

    let n_u32 = n as u32;
    let kl_u32 = kl as u32;
    let ku_u32 = ku as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&ab_ptr);
    builder.arg(&b_ptr);
    builder.arg(&x_ptr);
    builder.arg(&work_ptr);
    builder.arg(&n_u32);
    builder.arg(&kl_u32);
    builder.arg(&ku_u32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA banded_solve kernel launch failed: {:?}", e)))?;

    Ok(())
}
