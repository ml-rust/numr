//! SIMD-accelerated clamp operation
//!
//! clamp(x, min, max) = min(max(x, min), max)
//!
//! # SIMD Approach
//!
//! - Broadcast min and max values to vectors
//! - Use SIMD max then min operations

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

mod dispatch;
mod half;
mod scalar;

pub use dispatch::{clamp_f32, clamp_f64};
#[cfg(feature = "f16")]
pub use half::{clamp_bf16, clamp_f16};
