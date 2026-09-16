//! Matrix multiplication kernels.
//!
//! This module provides matrix multiplication with automatic SIMD dispatch.
//! On x86-64, f32 and f64 matmuls use AVX-512 or AVX2+FMA when available.
//!
//! # Accumulator width
//!
//! A dot product of length `k` outgrows the element type for every float
//! narrower than F32 and for every integer dtype, so those accumulate in a
//! wider type and narrow once per output element. See
//! [`crate::runtime::cpu::kernels::wide_acc`] for why the accumulators are f32
//! and i128, and why integer narrowing saturates. Output dtypes are unchanged,
//! except through the `*_wide_kernel` entry points, which store the F32
//! accumulator of a half matmul instead of narrowing it (`matmul_wide`).

mod bt;
mod dot;
mod gemv;
mod half_batch;
mod kernel;
mod wide;

pub use bt::{matmul_bt_kernel, matmul_bt_matches_contiguous};
pub use gemv::{gemv_bt_kernel, gemv_bt_wide_kernel};
pub use kernel::{matmul_bias_kernel, matmul_kernel};
pub use wide::matmul_wide_kernel;
