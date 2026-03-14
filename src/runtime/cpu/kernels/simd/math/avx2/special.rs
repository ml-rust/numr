//! AVX2 special function implementations (rsqrt, cbrt)
//!
//! # Safety
//!
//! All functions require AVX2 and FMA CPU features.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::exp_log::{exp_f64, log_f64};

// ============================================================================
// Additional transcendental functions
// ============================================================================

/// Fast SIMD rsqrt (1/sqrt(x)) for f32 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn rsqrt_f32(x: __m256) -> __m256 {
    // Use Newton-Raphson refinement on the fast approximation
    let approx = _mm256_rsqrt_ps(x);
    let half = _mm256_set1_ps(0.5);
    let three = _mm256_set1_ps(3.0);
    // One Newton-Raphson iteration: y = 0.5 * y * (3 - x * y * y)
    let x_approx2 = _mm256_mul_ps(x, _mm256_mul_ps(approx, approx));
    let factor = _mm256_sub_ps(three, x_approx2);
    _mm256_mul_ps(half, _mm256_mul_ps(approx, factor))
}

/// Fast SIMD rsqrt (1/sqrt(x)) for f64 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn rsqrt_f64(x: __m256d) -> __m256d {
    let sqrt_x = _mm256_sqrt_pd(x);
    _mm256_div_pd(_mm256_set1_pd(1.0), sqrt_x)
}

/// Fast SIMD cbrt (cube root) for f32 using AVX2
/// Uses Halley's method for refinement
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cbrt_f32(x: __m256) -> __m256 {
    // Handle sign separately
    let sign_mask = _mm256_set1_ps(-0.0);
    let sign = _mm256_and_ps(x, sign_mask);
    let abs_x = _mm256_andnot_ps(sign_mask, x);

    // Initial approximation using bit manipulation
    // cbrt(x) ≈ 2^(log2(x)/3) via IEEE 754
    let one_third = _mm256_set1_ps(1.0 / 3.0);
    let bias = _mm256_set1_ps(127.0);

    // Extract exponent: e = floor(log2(|x|))
    let xi = _mm256_castps_si256(abs_x);
    let exp_bits = _mm256_srli_epi32::<23>(xi);
    let exp_f = _mm256_cvtepi32_ps(_mm256_sub_epi32(exp_bits, _mm256_set1_epi32(127)));

    // Initial guess: 2^(e/3)
    let new_exp = _mm256_mul_ps(exp_f, one_third);
    let new_exp_i = _mm256_cvtps_epi32(_mm256_add_ps(new_exp, bias));
    let guess = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(new_exp_i));

    // Newton-Raphson iteration: y = y * (2*y^3 + x) / (2*x + y^3)
    // Simplified: y = (2*y + x/y^2) / 3
    let two = _mm256_set1_ps(2.0);
    let three = _mm256_set1_ps(3.0);

    let y = guess;
    let y2 = _mm256_mul_ps(y, y);
    let y_new = _mm256_div_ps(_mm256_fmadd_ps(two, y, _mm256_div_ps(abs_x, y2)), three);

    // One more iteration
    let y2 = _mm256_mul_ps(y_new, y_new);
    let result = _mm256_div_ps(_mm256_fmadd_ps(two, y_new, _mm256_div_ps(abs_x, y2)), three);

    // Restore sign
    _mm256_or_ps(result, sign)
}

/// Fast SIMD cbrt (cube root) for f64 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cbrt_f64(x: __m256d) -> __m256d {
    let sign_mask = _mm256_set1_pd(-0.0);
    let sign = _mm256_and_pd(x, sign_mask);
    let abs_x = _mm256_andnot_pd(sign_mask, x);

    let one_third = _mm256_set1_pd(1.0 / 3.0);

    // Initial guess: cbrt(x) ≈ exp(log(x) / 3)
    let log_x = log_f64(abs_x);
    let guess = exp_f64(_mm256_mul_pd(log_x, one_third));

    let two = _mm256_set1_pd(2.0);
    let three = _mm256_set1_pd(3.0);

    let y = guess;
    let y2 = _mm256_mul_pd(y, y);
    let y_new = _mm256_div_pd(_mm256_fmadd_pd(two, y, _mm256_div_pd(abs_x, y2)), three);

    let y2 = _mm256_mul_pd(y_new, y_new);
    let result = _mm256_div_pd(_mm256_fmadd_pd(two, y_new, _mm256_div_pd(abs_x, y2)), three);

    _mm256_or_pd(result, sign)
}
