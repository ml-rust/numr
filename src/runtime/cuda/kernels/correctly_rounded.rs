//! Launcher for the correctly rounded F64 kernels.

use std::sync::Arc;

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, launch_config,
};
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;

/// Module holding the correctly rounded kernels.
pub use super::loader::kernel_names::CORRECTLY_ROUNDED_MODULE;

/// The function a correctly rounded kernel applies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrKernel {
    /// `e^x`.
    Exp,
    /// `ln(x)`.
    Log,
    /// `ln(1 + x)`.
    Log1p,
    /// `acos(x)`.
    Acos,
    /// `x^exponent`.
    PowScalar(f64),
}

impl CrKernel {
    /// Kernel name in `correctly_rounded.cu`.
    fn name(self) -> &'static str {
        match self {
            CrKernel::Exp => "cr_exp_f64",
            CrKernel::Log => "cr_log_f64",
            CrKernel::Log1p => "cr_log1p_f64",
            CrKernel::Acos => "cr_acos_f64",
            CrKernel::PowScalar(_) => "cr_pow_scalar_f64",
        }
    }
}

/// Launch `kernel` over `lanes` F64 elements.
///
/// Lane `i` reads input element `i * step` and writes output element `i`.
///
/// # Safety
///
/// Both addresses point at device memory on the stream's device. The input holds
/// `(lanes - 1) * step + 1` F64 elements, and the output holds `lanes` F64 elements that
/// do not overlap the input.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_correctly_rounded(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    kernel: CrKernel,
    input_ptr: u64,
    step: u32,
    output_ptr: u64,
    lanes: u32,
) -> Result<()> {
    if lanes == 0 {
        return Ok(());
    }
    let name = kernel.name();
    let grid = elementwise_launch_config(lanes as usize)?;
    let module = get_or_load_module(context, device_index, CORRECTLY_ROUNDED_MODULE)?;
    let function = get_kernel_function(&module, name)?;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&input_ptr);
    builder.arg(&step);
    let exponent_bits = match kernel {
        CrKernel::PowScalar(exponent) => Some(exponent.to_bits()),
        _ => None,
    };
    if let Some(bits) = exponent_bits.as_ref() {
        builder.arg(bits);
    }
    builder.arg(&output_ptr);
    builder.arg(&lanes);
    unsafe {
        builder
            .launch(launch_config(grid, (BLOCK_SIZE, 1, 1), 0))
            .map_err(|error| {
                Error::Internal(format!(
                    "CUDA {name} launch failed for {lanes} lanes: {error:?}. Check the CUDA device state."
                ))
            })?;
    }
    Ok(())
}
