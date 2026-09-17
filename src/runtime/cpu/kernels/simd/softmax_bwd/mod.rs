//! SIMD-accelerated softmax backward operation.
//!
//! Computes: d_input[i] = output[i] * (grad[i] - dot)
//! where dot = sum(grad * output) along the softmax dimension.
//!
//! Fused 2-pass kernel:
//! - Pass 1: SIMD dot product (grad * output, reduced to scalar)
//! - Pass 2: SIMD elementwise output * (grad - dot)

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

mod dispatch;
mod half;
mod scalar;

pub use dispatch::{softmax_bwd_f32, softmax_bwd_f64};
#[cfg(feature = "f16")]
pub use half::{softmax_bwd_bf16, softmax_bwd_f16};
