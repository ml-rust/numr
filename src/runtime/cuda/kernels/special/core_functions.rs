//! CUDA kernel launchers for core special functions (error, gamma, beta)

use super::helpers::{launch_binary_special, launch_ternary_special, launch_unary_special};
use crate::dtype::DType;
use crate::error::Result;
use cudarc::driver::{CudaContext, CudaStream};
use std::sync::Arc;

// ============================================================================
// Error Functions
// ============================================================================

/// Launch erf kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_erf(
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
            "erf",
            "erf (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch erfc kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_erfc(
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
            "erfc",
            "erfc (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch erfinv kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_erfinv(
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
            "erfinv",
            "erfinv (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

// ============================================================================
// Gamma Functions
// ============================================================================

/// Launch gamma kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_gamma(
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
            "gamma",
            "gamma (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch lgamma kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_lgamma(
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
            "lgamma",
            "lgamma (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch digamma kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_digamma(
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
            "digamma",
            "digamma (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

// ============================================================================
// Beta Functions
// ============================================================================

/// Launch beta kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_beta(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_binary_special(
            ctx,
            stream,
            device_index,
            dtype,
            "beta",
            "beta (requires F32 or F64)",
            a_ptr,
            b_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch gammainc kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_gammainc(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_binary_special(
            ctx,
            stream,
            device_index,
            dtype,
            "gammainc",
            "gammainc (requires F32 or F64)",
            a_ptr,
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch gammaincc kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_gammaincc(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_binary_special(
            ctx,
            stream,
            device_index,
            dtype,
            "gammaincc",
            "gammaincc (requires F32 or F64)",
            a_ptr,
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch betainc kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_betainc(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_ternary_special(
            ctx,
            stream,
            device_index,
            dtype,
            "betainc",
            "betainc (requires F32 or F64)",
            a_ptr,
            b_ptr,
            x_ptr,
            out_ptr,
            numel,
        )
    }
}
