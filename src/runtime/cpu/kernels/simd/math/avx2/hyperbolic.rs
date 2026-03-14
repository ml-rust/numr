//! AVX2 hyperbolic function implementations (tanh, sinh, cosh, asinh, acosh, atanh)
//!
//! # Safety
//!
//! All functions require AVX2 and FMA CPU features.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::exp_log::{exp_f32, exp_f64, log_f32, log_f64};

// ============================================================================
// Hyperbolic tangent: tanh(x)
// ============================================================================

/// Fast SIMD tanh approximation for f32 using AVX2+FMA
///
/// Algorithm: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn tanh_f32(x: __m256) -> __m256 {
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);

    let exp2x = exp_f32(_mm256_mul_ps(two, x));
    let num = _mm256_sub_ps(exp2x, one);
    let den = _mm256_add_ps(exp2x, one);

    _mm256_div_ps(num, den)
}

/// Fast SIMD tanh approximation for f64 using AVX2+FMA
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn tanh_f64(x: __m256d) -> __m256d {
    let two = _mm256_set1_pd(2.0);
    let one = _mm256_set1_pd(1.0);

    let exp2x = exp_f64(_mm256_mul_pd(two, x));
    let num = _mm256_sub_pd(exp2x, one);
    let den = _mm256_add_pd(exp2x, one);

    _mm256_div_pd(num, den)
}

// ============================================================================
// Hyperbolic sine and cosine: sinh(x), cosh(x)
// ============================================================================

/// Fast SIMD sinh for f32 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sinh_f32(x: __m256) -> __m256 {
    // sinh(x) = (exp(x) - exp(-x)) / 2
    let half = _mm256_set1_ps(0.5);
    let exp_x = exp_f32(x);
    let exp_neg_x = exp_f32(_mm256_sub_ps(_mm256_setzero_ps(), x));
    _mm256_mul_ps(half, _mm256_sub_ps(exp_x, exp_neg_x))
}

/// Fast SIMD sinh for f64 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sinh_f64(x: __m256d) -> __m256d {
    let half = _mm256_set1_pd(0.5);
    let exp_x = exp_f64(x);
    let exp_neg_x = exp_f64(_mm256_sub_pd(_mm256_setzero_pd(), x));
    _mm256_mul_pd(half, _mm256_sub_pd(exp_x, exp_neg_x))
}

/// Fast SIMD cosh for f32 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cosh_f32(x: __m256) -> __m256 {
    // cosh(x) = (exp(x) + exp(-x)) / 2
    let half = _mm256_set1_ps(0.5);
    let exp_x = exp_f32(x);
    let exp_neg_x = exp_f32(_mm256_sub_ps(_mm256_setzero_ps(), x));
    _mm256_mul_ps(half, _mm256_add_ps(exp_x, exp_neg_x))
}

/// Fast SIMD cosh for f64 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cosh_f64(x: __m256d) -> __m256d {
    let half = _mm256_set1_pd(0.5);
    let exp_x = exp_f64(x);
    let exp_neg_x = exp_f64(_mm256_sub_pd(_mm256_setzero_pd(), x));
    _mm256_mul_pd(half, _mm256_add_pd(exp_x, exp_neg_x))
}

// ============================================================================
// Inverse hyperbolic functions: asinh, acosh, atanh
// ============================================================================

/// Fast SIMD asinh for f32 using AVX2
/// asinh(x) = log(x + sqrt(x^2 + 1))
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn asinh_f32(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let x2 = _mm256_mul_ps(x, x);
    let sqrt_term = _mm256_sqrt_ps(_mm256_add_ps(x2, one));
    log_f32(_mm256_add_ps(x, sqrt_term))
}

/// Fast SIMD asinh for f64 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn asinh_f64(x: __m256d) -> __m256d {
    let one = _mm256_set1_pd(1.0);
    let x2 = _mm256_mul_pd(x, x);
    let sqrt_term = _mm256_sqrt_pd(_mm256_add_pd(x2, one));
    log_f64(_mm256_add_pd(x, sqrt_term))
}

/// Fast SIMD acosh for f32 using AVX2
/// acosh(x) = log(x + sqrt(x^2 - 1)) for x >= 1
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn acosh_f32(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let x2 = _mm256_mul_ps(x, x);
    let sqrt_term = _mm256_sqrt_ps(_mm256_sub_ps(x2, one));
    log_f32(_mm256_add_ps(x, sqrt_term))
}

/// Fast SIMD acosh for f64 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn acosh_f64(x: __m256d) -> __m256d {
    let one = _mm256_set1_pd(1.0);
    let x2 = _mm256_mul_pd(x, x);
    let sqrt_term = _mm256_sqrt_pd(_mm256_sub_pd(x2, one));
    log_f64(_mm256_add_pd(x, sqrt_term))
}

/// Fast SIMD atanh for f32 using AVX2
/// atanh(x) = 0.5 * log((1 + x) / (1 - x)) for |x| < 1
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn atanh_f32(x: __m256) -> __m256 {
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let one_plus_x = _mm256_add_ps(one, x);
    let one_minus_x = _mm256_sub_ps(one, x);
    let ratio = _mm256_div_ps(one_plus_x, one_minus_x);
    _mm256_mul_ps(half, log_f32(ratio))
}

/// Fast SIMD atanh for f64 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn atanh_f64(x: __m256d) -> __m256d {
    let half = _mm256_set1_pd(0.5);
    let one = _mm256_set1_pd(1.0);
    let one_plus_x = _mm256_add_pd(one, x);
    let one_minus_x = _mm256_sub_pd(one, x);
    let ratio = _mm256_div_pd(one_plus_x, one_minus_x);
    _mm256_mul_pd(half, log_f64(ratio))
}
