//! SIMD-optimized cumulative operations dispatch
//!
//! Provides AVX2/AVX-512 accelerated cumsum and cumprod strided kernels.
//! The strided kernels vectorize over the inner_size dimension, where each
//! SIMD lane maintains its own independent accumulator.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

mod dispatch;
mod half;
mod scalar;

pub use dispatch::{
    cumprod_strided_f32, cumprod_strided_f64, cumsum_strided_f32, cumsum_strided_f64,
};
#[cfg(feature = "f16")]
pub use half::{
    cumprod_strided_bf16, cumprod_strided_f16, cumsum_strided_bf16, cumsum_strided_f16,
};
