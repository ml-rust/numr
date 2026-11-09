//! SIMD-accelerated logsumexp operation
//!
//! Logsumexp: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
//! Commonly used in attention mechanisms and probability computations.
//!
//! # SIMD Optimizations
//!
//! - SIMD max-reduce for finding maximum
//! - SIMD exp computation (vectorized polynomial approximation)
//! - SIMD sum-reduce for accumulation

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

mod dispatch;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod limits;
mod scalar;

#[cfg(feature = "f16")]
mod half;

pub use dispatch::{logsumexp_f32, logsumexp_f64};

#[cfg(feature = "f16")]
pub use half::{logsumexp_bf16, logsumexp_f16};

#[cfg(test)]
mod limits_tests;
