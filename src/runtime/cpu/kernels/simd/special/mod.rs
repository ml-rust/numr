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

use super::{SimdLevel, detect_simd};

/// Minimum elements to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

// ============================================================================
// Dispatch Macros - Eliminate duplication across special functions
// ============================================================================

/// Generate SIMD dispatch function for f32 with architecture-specific backends
macro_rules! impl_simd_dispatch_f32 {
    ($fn_name:ident, $scalar_fn:ident) => {
        #[inline]
        pub unsafe fn $fn_name(input: *const f32, output: *mut f32, len: usize) {
            let level = detect_simd();

            if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
                $scalar_fn(input, output, len);
                return;
            }

            #[cfg(target_arch = "x86_64")]
            match level {
                SimdLevel::Avx512 => avx512::$fn_name(input, output, len),
                SimdLevel::Avx2Fma => avx2::$fn_name(input, output, len),
                _ => $scalar_fn(input, output, len),
            }

            #[cfg(target_arch = "aarch64")]
            match level {
                SimdLevel::Neon | SimdLevel::NeonFp16 => {
                    aarch64::neon::$fn_name(input, output, len)
                }
                _ => $scalar_fn(input, output, len),
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            $scalar_fn(input, output, len);
        }
    };
}

/// Generate SIMD dispatch function for f64 with architecture-specific backends
macro_rules! impl_simd_dispatch_f64 {
    ($fn_name:ident, $scalar_fn:ident) => {
        #[inline]
        pub unsafe fn $fn_name(input: *const f64, output: *mut f64, len: usize) {
            let level = detect_simd();

            if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
                $scalar_fn(input, output, len);
                return;
            }

            #[cfg(target_arch = "x86_64")]
            match level {
                SimdLevel::Avx512 => avx512::$fn_name(input, output, len),
                SimdLevel::Avx2Fma => avx2::$fn_name(input, output, len),
                _ => $scalar_fn(input, output, len),
            }

            #[cfg(target_arch = "aarch64")]
            match level {
                SimdLevel::Neon | SimdLevel::NeonFp16 => {
                    aarch64::neon::$fn_name(input, output, len)
                }
                _ => $scalar_fn(input, output, len),
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            $scalar_fn(input, output, len);
        }
    };
}

/// Generate both f32 and f64 dispatch functions
macro_rules! impl_simd_dispatch {
    ($base_name:ident) => {
        paste::paste! {
            impl_simd_dispatch_f32!([<$base_name _f32>], [<$base_name _scalar_f32>]);
            impl_simd_dispatch_f64!([<$base_name _f64>], [<$base_name _scalar_f64>]);
        }
    };
}

/// Generate scalar fallback pair (f32 + f64) from a scalar function
macro_rules! impl_scalar_fallback {
    ($base_name:ident, $scalar_fn:path) => {
        paste::paste! {
            #[inline]
            unsafe fn [<$base_name _scalar_f32>](input: *const f32, output: *mut f32, len: usize) {
                for i in 0..len {
                    let x = *input.add(i);
                    *output.add(i) = $scalar_fn(x as f64) as f32;
                }
            }

            #[inline]
            unsafe fn [<$base_name _scalar_f64>](input: *const f64, output: *mut f64, len: usize) {
                for i in 0..len {
                    *output.add(i) = $scalar_fn(*input.add(i));
                }
            }
        }
    };
}

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

// ============================================================================
// Scalar Fallbacks - Import and generate typed wrappers
// ============================================================================

use crate::algorithm::special::scalar::{
    bessel_i0_scalar, bessel_i1_scalar, bessel_j0_scalar, bessel_j1_scalar, digamma_scalar,
    erf_scalar, erfc_scalar, gamma_scalar, lgamma_scalar,
};

impl_scalar_fallback!(erf, erf_scalar);
impl_scalar_fallback!(erfc, erfc_scalar);
impl_scalar_fallback!(bessel_j0, bessel_j0_scalar);
impl_scalar_fallback!(bessel_j1, bessel_j1_scalar);
impl_scalar_fallback!(bessel_i0, bessel_i0_scalar);
impl_scalar_fallback!(bessel_i1, bessel_i1_scalar);
impl_scalar_fallback!(gamma, gamma_scalar);
impl_scalar_fallback!(lgamma, lgamma_scalar);
impl_scalar_fallback!(digamma, digamma_scalar);

// ============================================================================
// SIMD Dispatch Functions - Error Functions
// ============================================================================

impl_simd_dispatch!(erf);
impl_simd_dispatch!(erfc);

// ============================================================================
// SIMD Dispatch Functions - Bessel Functions
// ============================================================================

impl_simd_dispatch!(bessel_j0);
impl_simd_dispatch!(bessel_j1);
impl_simd_dispatch!(bessel_i0);
impl_simd_dispatch!(bessel_i1);

// ============================================================================
// Gamma Functions - Scalar only (no SIMD implementation yet)
// ============================================================================

impl_scalar_only!(gamma);
impl_scalar_only!(lgamma);
impl_scalar_only!(digamma);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf_f32() {
        let input: Vec<f32> = (0..128).map(|x| (x as f32) * 0.05 - 3.0).collect();
        let mut out_simd = vec![0.0f32; 128];
        let mut out_scalar = vec![0.0f32; 128];

        unsafe {
            erf_f32(input.as_ptr(), out_simd.as_mut_ptr(), 128);
            erf_scalar_f32(input.as_ptr(), out_scalar.as_mut_ptr(), 128);
        }

        for i in 0..128 {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            assert!(
                diff < 1e-5,
                "erf mismatch at {}: SIMD={} scalar={} diff={}",
                i,
                out_simd[i],
                out_scalar[i],
                diff
            );
        }
    }

    #[test]
    fn test_bessel_j0_f32() {
        let input: Vec<f32> = (0..128).map(|x| (x as f32) * 0.2).collect();
        let mut out_simd = vec![0.0f32; 128];
        let mut out_scalar = vec![0.0f32; 128];

        unsafe {
            bessel_j0_f32(input.as_ptr(), out_simd.as_mut_ptr(), 128);
            bessel_j0_scalar_f32(input.as_ptr(), out_scalar.as_mut_ptr(), 128);
        }

        for i in 0..128 {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            let rel_err = if out_scalar[i].abs() > 1e-6 {
                diff / out_scalar[i].abs()
            } else {
                diff
            };
            assert!(
                rel_err < 1e-4,
                "bessel_j0 mismatch at {}: SIMD={} scalar={} rel_err={}",
                i,
                out_simd[i],
                out_scalar[i],
                rel_err
            );
        }
    }

    #[test]
    fn test_bessel_i0_f32() {
        // Test for modest arguments to avoid overflow
        let input: Vec<f32> = (0..128).map(|x| (x as f32) * 0.1).collect();
        let mut out_simd = vec![0.0f32; 128];
        let mut out_scalar = vec![0.0f32; 128];

        unsafe {
            bessel_i0_f32(input.as_ptr(), out_simd.as_mut_ptr(), 128);
            bessel_i0_scalar_f32(input.as_ptr(), out_scalar.as_mut_ptr(), 128);
        }

        for i in 0..128 {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            let rel_err = if out_scalar[i].abs() > 1e-6 {
                diff / out_scalar[i].abs()
            } else {
                diff
            };
            assert!(
                rel_err < 1e-4,
                "bessel_i0 mismatch at {}: SIMD={} scalar={} rel_err={}",
                i,
                out_simd[i],
                out_scalar[i],
                rel_err
            );
        }
    }
}
