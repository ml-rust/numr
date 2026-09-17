//! SIMD-accelerated scalar operations
//!
//! This module provides AVX2 and AVX-512 implementations for tensor-scalar operations.
//!
//! # SIMD Support
//!
//! Operations with SIMD fast paths:
//! - Add, Sub, Mul, Div, Max, Min (with scalar)
//!
//! Operations using scalar fallback:
//! - Pow (requires libm, no direct SIMD instruction)

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

mod dispatch;
mod half;
mod scalar;

pub use dispatch::{rsub_scalar_f32, rsub_scalar_f64, scalar_f32, scalar_f64};
#[cfg(feature = "f16")]
pub use half::{rsub_scalar_bf16, rsub_scalar_f16, scalar_bf16, scalar_f16};
