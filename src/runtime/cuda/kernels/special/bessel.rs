//! CUDA kernel launchers for Bessel functions (J0, J1, Y0, Y1, I0, I1, K0, K1)

use super::helpers::launch_unary_special;
use crate::dtype::DType;
use crate::error::Result;
use cudarc::driver::{CudaContext, CudaStream};
use std::sync::Arc;

/// Launch bessel_j0 kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_bessel_j0(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_special(
            ctx,
            stream,
            device_index,
            dtype,
            "bessel_j0",
            "bessel_j0 (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch bessel_j1 kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_bessel_j1(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_special(
            ctx,
            stream,
            device_index,
            dtype,
            "bessel_j1",
            "bessel_j1 (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch bessel_y0 kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_bessel_y0(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_special(
            ctx,
            stream,
            device_index,
            dtype,
            "bessel_y0",
            "bessel_y0 (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch bessel_y1 kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_bessel_y1(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_special(
            ctx,
            stream,
            device_index,
            dtype,
            "bessel_y1",
            "bessel_y1 (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch bessel_i0 kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_bessel_i0(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_special(
            ctx,
            stream,
            device_index,
            dtype,
            "bessel_i0",
            "bessel_i0 (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch bessel_i1 kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_bessel_i1(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_special(
            ctx,
            stream,
            device_index,
            dtype,
            "bessel_i1",
            "bessel_i1 (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch bessel_k0 kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_bessel_k0(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_special(
            ctx,
            stream,
            device_index,
            dtype,
            "bessel_k0",
            "bessel_k0 (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch bessel_k1 kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_bessel_k1(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_special(
            ctx,
            stream,
            device_index,
            dtype,
            "bessel_k1",
            "bessel_k1 (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}
