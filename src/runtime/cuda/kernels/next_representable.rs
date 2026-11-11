//! Raw integer CUDA launchers for adjacent IEEE encodings.

use std::sync::Arc;

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;

/// Module containing integer-only binary32 and binary64 neighbor kernels.
pub use super::loader::kernel_names::NEXT_REPRESENTABLE_MODULE;

/// Launch an integer-only neighbor kernel for contiguous tensors.
///
/// # Safety
///
/// Both pointers reference nonoverlapping, aligned allocations on the stream's device.
/// Each allocation contains at least `numel` elements of `dtype`.
pub unsafe fn launch_next_representable(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    upward: bool,
    input_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    let op = if upward { "next_up" } else { "next_down" };
    let name = match (dtype, upward) {
        (DType::F32, true) => "next_up_f32",
        (DType::F32, false) => "next_down_f32",
        (DType::F64, true) => "next_up_f64",
        (DType::F64, false) => "next_down_f64",
        _ => return Err(Error::UnsupportedDType { dtype, op }),
    };
    let invalid = |reason: String| Error::InvalidArgument {
        arg: "numel",
        reason,
    };
    let count = u32::try_from(numel).map_err(|_| {
        invalid(format!(
            "CUDA {op} count {numel} exceeds u32::MAX. Reduce the tensor size."
        ))
    })?;
    if count == 0 {
        return Ok(());
    }
    let bytes = numel.checked_mul(dtype.size_in_bytes()).ok_or_else(|| {
        invalid(format!(
            "CUDA {op} count {numel} overflows byte addressing. Reduce the tensor size."
        ))
    })?;
    let last_byte = u64::try_from(bytes - 1).map_err(|_| {
        invalid(format!(
            "CUDA {op} byte count {bytes} exceeds u64. Reduce the tensor size."
        ))
    })?;
    if input_ptr.checked_add(last_byte).is_none() || output_ptr.checked_add(last_byte).is_none() {
        return Err(invalid(format!(
            "CUDA {op} device address overflows u64 for {numel} elements. Use an in-bounds allocation."
        )));
    }
    let grid = elementwise_launch_config(numel)?;
    let module = get_or_load_module(context, device_index, NEXT_REPRESENTABLE_MODULE)?;
    let function = get_kernel_function(&module, name)?;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&count);
    unsafe {
        builder
            .launch(launch_config(grid, (BLOCK_SIZE, 1, 1), 0))
            .map_err(|error| {
                Error::Internal(format!(
                    "CUDA {name} launch failed for {numel} elements: {error:?}. Check the CUDA device state."
                ))
            })?;
    }
    Ok(())
}
