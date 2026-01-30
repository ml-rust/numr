//! Generic CUDA launcher helpers for special mathematical functions
//!
//! Provides reusable launcher infrastructure for unary, binary, and ternary
//! special function kernels.

use super::super::loader::{
    elementwise_launch_config, get_kernel_function, get_or_load_module, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use cudarc::driver::{CudaContext, CudaStream, PushKernelArg};
use std::sync::Arc;

pub(crate) const SPECIAL_MODULE: &str = "special";

/// Get kernel name with dtype suffix for special functions
pub(crate) fn special_kernel_name(
    base: &str,
    dtype: DType,
    op_name: &'static str,
) -> Result<String> {
    let suffix = match dtype {
        DType::F32 => "f32",
        DType::F64 => "f64",
        _ => {
            return Err(Error::UnsupportedDType { dtype, op: op_name });
        }
    };
    Ok(format!("{}_{}", base, suffix))
}

/// Generic launcher for unary special functions (1 input -> 1 output)
///
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub(crate) unsafe fn launch_unary_special(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    kernel_base: &str,
    op_name: &'static str,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = special_kernel_name(kernel_base, dtype, op_name)?;
    let module = get_or_load_module(ctx, device_index, SPECIAL_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel);
    let cfg = launch_config(grid, (256, 1, 1), 0);
    let n = numel as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&x_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Generic launcher for binary special functions (2 inputs -> 1 output)
///
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub(crate) unsafe fn launch_binary_special(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    kernel_base: &str,
    op_name: &'static str,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = special_kernel_name(kernel_base, dtype, op_name)?;
    let module = get_or_load_module(ctx, device_index, SPECIAL_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel);
    let cfg = launch_config(grid, (256, 1, 1), 0);
    let n = numel as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Generic launcher for ternary special functions (3 inputs -> 1 output)
///
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub(crate) unsafe fn launch_ternary_special(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    kernel_base: &str,
    op_name: &'static str,
    a_ptr: u64,
    b_ptr: u64,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = special_kernel_name(kernel_base, dtype, op_name)?;
    let module = get_or_load_module(ctx, device_index, SPECIAL_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel);
    let cfg = launch_config(grid, (256, 1, 1), 0);
    let n = numel as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&x_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}
