//! Launcher for the single-pass interval and expansion kernels.

use std::sync::Arc;

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, launch_config,
};
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;

/// Module holding every interval and expansion kernel.
pub use super::loader::kernel_names::INTERVAL_MODULE;

/// One scalar kernel argument, in declaration order.
#[derive(Debug, Clone, Copy)]
pub enum IntervalArg {
    /// Device address.
    Ptr(u64),
    /// Element step, lane count, or flag.
    U32(u32),
    /// Operation code or switch.
    I32(i32),
    /// F32 constant.
    F32(f32),
    /// F64 constant.
    F64(f64),
}

/// Launch `name` from the interval module over `lanes` threads.
///
/// # Safety
///
/// `args` matches the kernel signature. Every address points at device memory on the stream's
/// device that holds the elements the kernel reads or writes for `lanes` lanes.
pub unsafe fn launch_interval(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    name: &str,
    args: &[IntervalArg],
    lanes: usize,
) -> Result<()> {
    if lanes == 0 {
        return Ok(());
    }
    let grid = elementwise_launch_config(lanes)?;
    let module = get_or_load_module(context, device_index, INTERVAL_MODULE)?;
    let function = get_kernel_function(&module, name)?;
    let mut builder = stream.launch_builder(&function);
    for arg in args {
        match arg {
            IntervalArg::Ptr(value) => builder.arg(value),
            IntervalArg::U32(value) => builder.arg(value),
            IntervalArg::I32(value) => builder.arg(value),
            IntervalArg::F32(value) => builder.arg(value),
            IntervalArg::F64(value) => builder.arg(value),
        };
    }
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
