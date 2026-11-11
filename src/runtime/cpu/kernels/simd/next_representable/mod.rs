//! Native integer SIMD paths and portable tails.

#[cfg(target_arch = "x86_64")]
/// AVX2 integer kernels.
pub mod avx2;
#[cfg(target_arch = "x86_64")]
/// AVX-512 integer kernels.
pub mod avx512;
/// Dispatch using an existing SIMD capability level.
pub mod dispatch;
#[cfg(target_arch = "aarch64")]
/// NEON integer kernels.
pub mod neon;
/// Portable integer stepping.
pub mod scalar;

pub use dispatch::{next_f32, next_f64};
