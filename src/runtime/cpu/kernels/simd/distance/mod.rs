//! SIMD per-pair reductions for distance metrics.
//!
//! Each kernel reduces one pair of contiguous `f32` or `f64` vectors to a sum.
//! The final metric formula (square root, cosine ratio) stays in
//! `kernels::distance::metrics`, so this module only owns the summation.
//!
//! # Architecture Support
//!
//! | Architecture | Instruction Set | Status                         |
//! |--------------|-----------------|--------------------------------|
//! | x86-64       | AVX-512         | Runs the AVX2+FMA kernels      |
//! | x86-64       | AVX2 + FMA      | Dedicated kernels              |
//! | ARM64        | NEON            | Runs the scalar kernels        |
//! | Any          | Scalar          | Same order as `metrics.rs`     |

mod dispatch;
mod scalar;
mod sums;
#[cfg(target_arch = "x86_64")]
mod x86_64;

pub use dispatch::{
    cosine_sums_f32, cosine_sums_f64, manhattan_f32, manhattan_f64, sqeuclidean_f32,
    sqeuclidean_f64,
};
