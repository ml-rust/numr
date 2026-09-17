//! SIMD-optimized index operations dispatch
//!
//! Provides AVX2/AVX-512 accelerated masked_fill and masked_select operations.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

mod dispatch;
mod generic;
mod scalar;

pub use dispatch::{
    masked_count, masked_fill_f32, masked_fill_f64, masked_select_f32, masked_select_f64,
};
