//! SIMD-accelerated fused activation-multiplication operations
//!
//! Provides vectorized implementations of fused activation * multiplication:
//! - silu_mul: (x / (1 + exp(-x))) * y
//! - gelu_mul: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) * y
//! - relu_mul: max(0, x) * y
//! - sigmoid_mul: (1 / (1 + exp(-x))) * y
//!
//! These operations take TWO inputs (a, b) and compute `activation(a) * b` in one pass,
//! reducing memory bandwidth compared to separate operations.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

mod dispatch;
mod half;
mod scalar;

pub use dispatch::*;
pub use half::*;
pub use scalar::*;
