//! Reduction operation kernels
//!
//! Provides reduction operations with automatic SIMD dispatch.
//! On x86-64, f32 and f64 operations use AVX-512 or AVX2 when available.

mod acc_kernel;
mod accumulator;
mod dispatch;
mod int_acc;
mod scalar;
mod special;

pub use accumulator::Accumulator;
pub use dispatch::{reduce_kernel, reduce_kernel_with_precision};
pub use special::{
    argmax_kernel, argmin_kernel, softmax_bwd_kernel, softmax_kernel, variance_kernel,
};
