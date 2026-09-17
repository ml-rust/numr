//! SIMD-accelerated conditional select (where) operation
//!
//! where(cond, x, y): out[i] = cond[i] ? x[i] : y[i]
//!
//! # SIMD Approach
//!
//! - Load condition bytes and expand to element-width masks
//! - Use SIMD blend operations based on mask

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

pub use dispatch::{where_f32, where_f64};
#[cfg(feature = "f16")]
pub use half::{where_bf16, where_f16};
