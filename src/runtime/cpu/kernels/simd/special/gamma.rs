//! Gamma family special functions (gamma, lgamma, digamma) — scalar only, no
//! SIMD implementation yet.
//!
//! Split out of `dispatch.rs` to keep that file under the kernel-file line
//! cap once its own tests were inlined.

use super::dispatch::impl_scalar_fallback;
use crate::algorithm::special::scalar::{digamma_scalar, gamma_scalar, lgamma_scalar};

/// Generate scalar-only dispatch (no SIMD implementation available)
macro_rules! impl_scalar_only {
    ($base_name:ident) => {
        paste::paste! {
            #[inline]
            pub unsafe fn [<$base_name _f32>](input: *const f32, output: *mut f32, len: usize) {
                [<$base_name _scalar_f32>](input, output, len);
            }

            #[inline]
            pub unsafe fn [<$base_name _f64>](input: *const f64, output: *mut f64, len: usize) {
                [<$base_name _scalar_f64>](input, output, len);
            }
        }
    };
}

impl_scalar_fallback!(gamma, gamma_scalar);
impl_scalar_fallback!(lgamma, lgamma_scalar);
impl_scalar_fallback!(digamma, digamma_scalar);

impl_scalar_only!(gamma);
impl_scalar_only!(lgamma);
impl_scalar_only!(digamma);

// F16/BF16 Wrappers via macros
half_unary!(gamma, gamma_f32);
half_unary!(lgamma, lgamma_f32);
half_unary!(digamma, digamma_f32);
