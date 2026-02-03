//! CUDA kernel launchers for extended special functions
//!
//! This module provides launchers for:
//! - Elliptic integrals: ellipk, ellipe
//! - Hypergeometric functions: hyp2f1, hyp1f1
//! - Airy functions: airy_ai, airy_bi
//! - Legendre functions: legendre_p, legendre_p_assoc
//! - Spherical harmonics: sph_harm
//! - Fresnel integrals: fresnel_s, fresnel_c

use super::helpers::{
    launch_binary_special_with_two_ints, launch_unary_special, launch_unary_special_with_2f64,
    launch_unary_special_with_3f64, launch_unary_special_with_int,
    launch_unary_special_with_two_ints,
};
use crate::dtype::DType;
use crate::error::Result;
use cudarc::driver::{CudaContext, CudaStream};
use std::sync::Arc;

// ============================================================================
// Elliptic Integrals
// ============================================================================

/// Launch complete elliptic integral K(m) kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_ellipk(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    m_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_special(
            ctx,
            stream,
            device_index,
            dtype,
            "ellipk",
            "ellipk (requires F32 or F64)",
            m_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch complete elliptic integral E(m) kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_ellipe(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    m_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_special(
            ctx,
            stream,
            device_index,
            dtype,
            "ellipe",
            "ellipe (requires F32 or F64)",
            m_ptr,
            out_ptr,
            numel,
        )
    }
}

// ============================================================================
// Hypergeometric Functions
// ============================================================================

/// Launch Gauss hypergeometric function 2F1(a,b;c;z) kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_hyp2f1(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a: f64,
    b: f64,
    c: f64,
    z_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_special_with_3f64(
            ctx,
            stream,
            device_index,
            dtype,
            "hyp2f1",
            "hyp2f1 (requires F32 or F64)",
            a,
            b,
            c,
            z_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch confluent hypergeometric function 1F1(a;b;z) kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_hyp1f1(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a: f64,
    b: f64,
    z_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_special_with_2f64(
            ctx,
            stream,
            device_index,
            dtype,
            "hyp1f1",
            "hyp1f1 (requires F32 or F64)",
            a,
            b,
            z_ptr,
            out_ptr,
            numel,
        )
    }
}

// ============================================================================
// Airy Functions
// ============================================================================

/// Launch Airy Ai function kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_airy_ai(
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
            "airy_ai",
            "airy_ai (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch Airy Bi function kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_airy_bi(
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
            "airy_bi",
            "airy_bi (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

// ============================================================================
// Legendre Functions
// ============================================================================

/// Launch Legendre polynomial P_n(x) kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_legendre_p(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    n: i32,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_special_with_int(
            ctx,
            stream,
            device_index,
            dtype,
            "legendre_p",
            "legendre_p (requires F32 or F64)",
            n,
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch associated Legendre function P_n^m(x) kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_legendre_p_assoc(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    n: i32,
    m: i32,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_special_with_two_ints(
            ctx,
            stream,
            device_index,
            dtype,
            "legendre_p_assoc",
            "legendre_p_assoc (requires F32 or F64)",
            n,
            m,
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

// ============================================================================
// Spherical Harmonics
// ============================================================================

/// Launch spherical harmonic Y_n^m(theta, phi) kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_sph_harm(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    n: i32,
    m: i32,
    theta_ptr: u64,
    phi_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_binary_special_with_two_ints(
            ctx,
            stream,
            device_index,
            dtype,
            "sph_harm",
            "sph_harm (requires F32 or F64)",
            n,
            m,
            theta_ptr,
            phi_ptr,
            out_ptr,
            numel,
        )
    }
}

// ============================================================================
// Fresnel Integrals
// ============================================================================

/// Launch Fresnel S integral kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_fresnel_s(
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
            "fresnel_s",
            "fresnel_s (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch Fresnel C integral kernel
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub unsafe fn launch_fresnel_c(
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
            "fresnel_c",
            "fresnel_c (requires F32 or F64)",
            x_ptr,
            out_ptr,
            numel,
        )
    }
}
