//! SIMD-accelerated softmax operation using the online softmax algorithm.
//!
//! Softmax is critical for attention mechanisms in transformers.
//! softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
//!
//! # Online Softmax Algorithm (2-pass)
//!
//! Instead of the traditional 3-pass approach (find max, compute exp+sum, normalize),
//! we use a 2-pass online algorithm:
//!
//! **Pass 1 (online max + sum):** For each element x[i], maintain running max `m` and
//! running sum `s`. When a new max is found, rescale the accumulated sum.
//!
//! **Pass 2 (normalize):** output[i] = exp(x[i] - m) / s
//!
//! This saves one full read+write pass over the output buffer compared to 3-pass.

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

pub use dispatch::{softmax_f32, softmax_f64};
#[cfg(feature = "f16")]
pub use half::{softmax_bf16, softmax_f16};
