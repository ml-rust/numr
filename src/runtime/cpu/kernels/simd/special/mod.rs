//! SIMD-accelerated special functions
//!
//! Provides vectorized implementations of error functions and Bessel functions.
//! Uses polynomial evaluation with FMA for accuracy and performance.
//!
//! # Supported Functions
//!
//! | Function  | Algorithm                        | Accuracy        |
//! |-----------|----------------------------------|-----------------|
//! | erf       | A&S 7.1.26 polynomial            | ~1e-7 rel err   |
//! | erfc      | 1 - erf(x)                       | ~1e-7 rel err   |
//! | bessel_j0 | Rational poly + asymptotic       | ~1e-8 rel err   |
//! | bessel_j1 | Rational poly + asymptotic       | ~1e-8 rel err   |
//! | bessel_i0 | Power series + asymptotic        | ~1e-8 rel err   |
//! | bessel_i1 | Power series + asymptotic        | ~1e-8 rel err   |

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

pub mod coefficients;

mod dispatch;
mod gamma;

#[cfg(feature = "f16")]
pub use dispatch::{
    bessel_i0_bf16, bessel_i0_f16, bessel_i1_bf16, bessel_i1_f16, bessel_j0_bf16, bessel_j0_f16,
    bessel_j1_bf16, bessel_j1_f16, erf_bf16, erf_f16, erfc_bf16, erfc_f16,
};
pub use dispatch::{
    bessel_i0_f32, bessel_i0_f64, bessel_i1_f32, bessel_i1_f64, bessel_j0_f32, bessel_j0_f64,
    bessel_j1_f32, bessel_j1_f64, erf_f32, erf_f64, erfc_f32, erfc_f64,
};
#[cfg(feature = "f16")]
pub use gamma::{digamma_bf16, digamma_f16, gamma_bf16, gamma_f16, lgamma_bf16, lgamma_f16};
pub use gamma::{digamma_f32, digamma_f64, gamma_f32, gamma_f64, lgamma_f32, lgamma_f64};
