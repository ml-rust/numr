//! SIMD-accelerated comparison operations
//!
//! This module provides AVX2 and AVX-512 implementations for element-wise
//! comparison operations (eq, ne, lt, le, gt, ge).
//!
//! # SIMD Approach
//!
//! - Use SIMD compare intrinsics to generate masks
//! - Blend between 1.0 and 0.0 vectors based on masks
//! - Output 1.0 for true, 0.0 for false (matching scalar behavior)

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

mod dispatch;
#[cfg(feature = "f16")]
mod half;
mod scalar;

pub use dispatch::{compare_f32, compare_f64};
#[cfg(feature = "f16")]
pub use half::{compare_bf16, compare_f16};
pub use scalar::{compare_scalar_f32, compare_scalar_f64};
