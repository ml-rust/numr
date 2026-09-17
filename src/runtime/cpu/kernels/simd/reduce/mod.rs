//! SIMD-accelerated reduction operations
//!
//! This module provides AVX2 and AVX-512 implementations for reduction operations.
//!
//! # SIMD Support
//!
//! Operations with SIMD fast paths:
//! - Sum, Max, Min, Prod
//!
//! Operations using scalar (no SIMD benefit or complex logic):
//! - Mean (uses Sum + division), All, Any

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

mod dispatch;
mod half;
mod scalar;

pub use dispatch::{reduce_f32, reduce_f64};
#[cfg(feature = "f16")]
pub use half::{reduce_bf16, reduce_f16};
