//! Utility CUDA kernel launchers
//!
//! Provides launchers for utility operations like fill (initialize tensor with constant value).
//!
//! Note: These launchers are prepared for future optimization of tensor creation methods
//! (zeros, ones, full_scalar) which currently use CPU-to-GPU copy. Once wired up, these
//! will allow direct GPU fill operations for better performance.

#![allow(dead_code)] // Prepared for future tensor creation optimization

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    kernel_names, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Value representation for fill operations.
///
/// This enum allows passing fill values of different types through a unified interface
/// while maintaining type safety at the kernel boundary.
#[derive(Debug, Clone, Copy)]
pub enum FillValue {
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    U8(u8),
}

impl FillValue {
    /// Create a FillValue from an f64, converting to the appropriate type for the given dtype.
    pub fn from_f64(value: f64, dtype: DType) -> Self {
        match dtype {
            DType::F32 => FillValue::F32(value as f32),
            DType::F64 => FillValue::F64(value),
            DType::I32 => FillValue::I32(value as i32),
            DType::I64 => FillValue::I64(value as i64),
            DType::U8 | DType::Bool => FillValue::U8(value as u8),
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => FillValue::F32(value as f32), // F16/BF16 kernels use f32 value
            #[cfg(feature = "fp8")]
            DType::FP8E4M3 | DType::FP8E5M2 => FillValue::F32(value as f32), // FP8 kernels use f32 value
            _ => FillValue::F64(value), // Default fallback
        }
    }

    /// Get the dtype this value corresponds to for kernel dispatch.
    fn kernel_dtype(&self) -> DType {
        match self {
            FillValue::F32(_) => DType::F32,
            FillValue::F64(_) => DType::F64,
            FillValue::I32(_) => DType::I32,
            FillValue::I64(_) => DType::I64,
            FillValue::U8(_) => DType::U8,
        }
    }
}

/// Launch a fill kernel for any supported dtype.
///
/// Fills the output tensor with a constant value. This is the unified entry point
/// that dispatches to the appropriate typed kernel.
///
/// # Safety
///
/// - `out_ptr` must be valid device memory with at least `numel` elements of the given dtype
/// - The `value` dtype must match the actual data type at `out_ptr`
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of the output tensor
/// * `value` - Value to fill with (will be converted to appropriate type)
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
///
/// # Example
///
/// ```ignore
/// // Fill with f32
/// unsafe {
///     launch_fill(ctx, stream, 0, DType::F32, FillValue::F32(1.0), ptr, 1024)?;
/// }
///
/// // Fill with automatic conversion from f64
/// unsafe {
///     launch_fill(ctx, stream, 0, DType::I32, FillValue::from_f64(42.0, DType::I32), ptr, 1024)?;
/// }
/// ```
pub unsafe fn launch_fill(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    _dtype: DType,
    value: FillValue,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::UTILITY_MODULE)?;
    let func_name = kernel_name("fill", value.kernel_dtype());
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel);
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    // Build and launch inside each match arm to ensure value lives long enough
    // SAFETY: All launch calls use valid kernel arguments with correct types
    let launch_result = match value {
        FillValue::F32(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        FillValue::F64(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        FillValue::I32(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        FillValue::I64(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        FillValue::U8(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
    };

    launch_result.map_err(|e| {
        Error::Internal(format!(
            "CUDA fill kernel '{}' launch failed: {:?}",
            func_name, e
        ))
    })?;

    Ok(())
}

/// Convenience function: Launch a fill kernel from an f64 value.
///
/// Automatically converts the f64 value to the appropriate type for the given dtype.
///
/// # Safety
///
/// Same requirements as [`launch_fill`].
pub unsafe fn launch_fill_with_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    value: f64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    // SAFETY: Caller must ensure out_ptr is valid device memory
    unsafe {
        launch_fill(
            context,
            stream,
            device_index,
            dtype,
            FillValue::from_f64(value, dtype),
            out_ptr,
            numel,
        )
    }
}
