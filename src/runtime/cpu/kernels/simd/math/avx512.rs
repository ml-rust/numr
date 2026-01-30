//! AVX-512 mathematical function implementations
//!
//! Provides vectorized transcendental functions using 512-bit registers.
//! All algorithms and coefficients are documented in `common.rs`.
//!
//! # Supported Functions
//!
//! | Function | f32 | f64 | Relative Error |
//! |----------|-----|-----|----------------|
//! | exp      | ✓   | ✓   | < 1e-6 / 1e-12 |
//! | tanh     | ✓   | ✓   | < 1e-6 / 1e-12 |
//! | log      | ✓   | ✓   | < 1e-6 / 1e-12 |
//! | sin      | ✓   | ✓   | < 1e-6 / 1e-10 |
//! | cos      | ✓   | ✓   | < 1e-6 / 1e-10 |
//! | tan      | ✓   | ✓   | < 2e-4 / 1e-4  |
//!
//! # Safety
//!
//! All functions require AVX-512F CPU feature.
//!
//! # Advantages over AVX2
//!
//! - Native 64-bit integer conversion (`_mm512_cvtpd_epi64`, `_mm512_cvtepi64_pd`)
//! - Masked operations for branchless conditionals
//! - Twice the throughput (16 f32 or 8 f64 per operation)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::common::{exp_coefficients, log_coefficients, tan_coefficients, trig_coefficients};

// ============================================================================
// Exponential function: exp(x)
// ============================================================================

/// Fast SIMD exp approximation for f32 using AVX-512
///
/// See `common::_EXP_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX-512F CPU feature.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn exp_f32(x: __m512) -> __m512 {
    use exp_coefficients::*;

    let log2e = _mm512_set1_ps(std::f32::consts::LOG2_E);
    let ln2 = _mm512_set1_ps(std::f32::consts::LN_2);

    let c0 = _mm512_set1_ps(C0_F32);
    let c1 = _mm512_set1_ps(C1_F32);
    let c2 = _mm512_set1_ps(C2_F32);
    let c3 = _mm512_set1_ps(C3_F32);
    let c4 = _mm512_set1_ps(C4_F32);
    let c5 = _mm512_set1_ps(C5_F32);
    let c6 = _mm512_set1_ps(C6_F32);

    // Clamp input to avoid overflow/underflow
    let x = _mm512_max_ps(x, _mm512_set1_ps(MIN_F32));
    let x = _mm512_min_ps(x, _mm512_set1_ps(MAX_F32));

    // y = x * log2(e)
    let y = _mm512_mul_ps(x, log2e);

    // n = round(y)
    let n = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(y);

    // f = y - n (fractional part)
    let f = _mm512_sub_ps(y, n);

    // r = f * ln(2)
    let r = _mm512_mul_ps(f, ln2);

    // Polynomial approximation
    let r2 = _mm512_mul_ps(r, r);
    let r3 = _mm512_mul_ps(r2, r);
    let r4 = _mm512_mul_ps(r2, r2);
    let r5 = _mm512_mul_ps(r4, r);
    let r6 = _mm512_mul_ps(r4, r2);

    let mut poly = c0;
    poly = _mm512_fmadd_ps(c1, r, poly);
    poly = _mm512_fmadd_ps(c2, r2, poly);
    poly = _mm512_fmadd_ps(c3, r3, poly);
    poly = _mm512_fmadd_ps(c4, r4, poly);
    poly = _mm512_fmadd_ps(c5, r5, poly);
    poly = _mm512_fmadd_ps(c6, r6, poly);

    // Compute 2^n using IEEE 754 bit manipulation
    let n_i32 = _mm512_cvtps_epi32(n);
    let bias = _mm512_set1_epi32(127);
    let exp_bits = _mm512_slli_epi32::<23>(_mm512_add_epi32(n_i32, bias));
    let pow2n = _mm512_castsi512_ps(exp_bits);

    _mm512_mul_ps(pow2n, poly)
}

/// Fast SIMD exp approximation for f64 using AVX-512
///
/// See `common::_EXP_ALGORITHM_DOC` for algorithm details.
///
/// # Note
/// Unlike AVX2, AVX-512 has native 64-bit integer conversion, so this
/// implementation is fully vectorized with no scalar operations.
///
/// # Safety
/// Requires AVX-512F CPU feature.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn exp_f64(x: __m512d) -> __m512d {
    use exp_coefficients::*;

    let log2e = _mm512_set1_pd(std::f64::consts::LOG2_E);
    let ln2 = _mm512_set1_pd(std::f64::consts::LN_2);

    let c0 = _mm512_set1_pd(C0_F64);
    let c1 = _mm512_set1_pd(C1_F64);
    let c2 = _mm512_set1_pd(C2_F64);
    let c3 = _mm512_set1_pd(C3_F64);
    let c4 = _mm512_set1_pd(C4_F64);
    let c5 = _mm512_set1_pd(C5_F64);
    let c6 = _mm512_set1_pd(C6_F64);

    // Clamp input
    let x = _mm512_max_pd(x, _mm512_set1_pd(MIN_F64));
    let x = _mm512_min_pd(x, _mm512_set1_pd(MAX_F64));

    let y = _mm512_mul_pd(x, log2e);
    let n = _mm512_roundscale_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(y);
    let f = _mm512_sub_pd(y, n);
    let r = _mm512_mul_pd(f, ln2);

    let r2 = _mm512_mul_pd(r, r);
    let r3 = _mm512_mul_pd(r2, r);
    let r4 = _mm512_mul_pd(r2, r2);
    let r5 = _mm512_mul_pd(r4, r);
    let r6 = _mm512_mul_pd(r4, r2);

    let mut poly = c0;
    poly = _mm512_fmadd_pd(c1, r, poly);
    poly = _mm512_fmadd_pd(c2, r2, poly);
    poly = _mm512_fmadd_pd(c3, r3, poly);
    poly = _mm512_fmadd_pd(c4, r4, poly);
    poly = _mm512_fmadd_pd(c5, r5, poly);
    poly = _mm512_fmadd_pd(c6, r6, poly);

    // AVX-512 has native 64-bit integer conversion
    let n_i64 = _mm512_cvtpd_epi64(n);
    let bias = _mm512_set1_epi64(1023);
    let exp_bits = _mm512_slli_epi64::<52>(_mm512_add_epi64(n_i64, bias));
    let pow2n = _mm512_castsi512_pd(exp_bits);

    _mm512_mul_pd(pow2n, poly)
}

// ============================================================================
// Hyperbolic tangent: tanh(x)
// ============================================================================

/// Fast SIMD tanh approximation for f32 using AVX-512
///
/// Algorithm: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
///
/// # Safety
/// Requires AVX-512F CPU feature.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn tanh_f32(x: __m512) -> __m512 {
    let two = _mm512_set1_ps(2.0);
    let one = _mm512_set1_ps(1.0);

    let exp2x = exp_f32(_mm512_mul_ps(two, x));
    let num = _mm512_sub_ps(exp2x, one);
    let den = _mm512_add_ps(exp2x, one);

    _mm512_div_ps(num, den)
}

/// Fast SIMD tanh approximation for f64 using AVX-512
///
/// # Safety
/// Requires AVX-512F CPU feature.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn tanh_f64(x: __m512d) -> __m512d {
    let two = _mm512_set1_pd(2.0);
    let one = _mm512_set1_pd(1.0);

    let exp2x = exp_f64(_mm512_mul_pd(two, x));
    let num = _mm512_sub_pd(exp2x, one);
    let den = _mm512_add_pd(exp2x, one);

    _mm512_div_pd(num, den)
}

// ============================================================================
// Natural logarithm: log(x)
// ============================================================================

/// Fast SIMD log approximation for f32 using AVX-512
///
/// See `common::_LOG_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX-512F CPU feature.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn log_f32(x: __m512) -> __m512 {
    use log_coefficients::*;

    let one = _mm512_set1_ps(1.0);
    let ln2 = _mm512_set1_ps(std::f32::consts::LN_2);
    let sqrt2 = _mm512_set1_ps(std::f32::consts::SQRT_2);
    let half = _mm512_set1_ps(0.5);

    let c1 = _mm512_set1_ps(C1_F32);
    let c2 = _mm512_set1_ps(C2_F32);
    let c3 = _mm512_set1_ps(C3_F32);
    let c4 = _mm512_set1_ps(C4_F32);
    let c5 = _mm512_set1_ps(C5_F32);
    let c6 = _mm512_set1_ps(C6_F32);
    let c7 = _mm512_set1_ps(C7_F32);

    // Extract exponent
    let x_bits = _mm512_castps_si512(x);
    let exp_raw = _mm512_srli_epi32::<23>(x_bits);
    let exp_unbiased = _mm512_sub_epi32(exp_raw, _mm512_set1_epi32(EXP_BIAS_F32));
    let mut n = _mm512_cvtepi32_ps(exp_unbiased);

    // Extract mantissa in [1, 2)
    let mantissa_mask = _mm512_set1_epi32(MANTISSA_MASK_F32);
    let exp_zero = _mm512_set1_epi32(EXP_ZERO_F32);
    let m_bits = _mm512_or_si512(_mm512_and_si512(x_bits, mantissa_mask), exp_zero);
    let mut m = _mm512_castsi512_ps(m_bits);

    // Normalize: if m > sqrt(2), divide by 2 and increment exponent
    // AVX-512 masked operations are branchless
    let need_adjust = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(m, sqrt2);
    m = _mm512_mask_mul_ps(m, need_adjust, m, half);
    n = _mm512_mask_add_ps(n, need_adjust, n, one);

    let f = _mm512_sub_ps(m, one);

    // Horner's method
    let mut poly = c7;
    poly = _mm512_fmadd_ps(poly, f, c6);
    poly = _mm512_fmadd_ps(poly, f, c5);
    poly = _mm512_fmadd_ps(poly, f, c4);
    poly = _mm512_fmadd_ps(poly, f, c3);
    poly = _mm512_fmadd_ps(poly, f, c2);
    poly = _mm512_fmadd_ps(poly, f, c1);
    poly = _mm512_mul_ps(poly, f);

    _mm512_fmadd_ps(n, ln2, poly)
}

/// Fast SIMD log approximation for f64 using AVX-512
///
/// See `common::_LOG_ALGORITHM_DOC` for algorithm details.
///
/// # Note
/// AVX-512 has native 64-bit integer operations, so this implementation
/// is fully vectorized with no scalar operations.
///
/// # Safety
/// Requires AVX-512F CPU feature.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn log_f64(x: __m512d) -> __m512d {
    use log_coefficients::*;

    let one = _mm512_set1_pd(1.0);
    let ln2 = _mm512_set1_pd(std::f64::consts::LN_2);
    let sqrt2 = _mm512_set1_pd(std::f64::consts::SQRT_2);
    let half = _mm512_set1_pd(0.5);

    let c1 = _mm512_set1_pd(C1_F64);
    let c2 = _mm512_set1_pd(C2_F64);
    let c3 = _mm512_set1_pd(C3_F64);
    let c4 = _mm512_set1_pd(C4_F64);
    let c5 = _mm512_set1_pd(C5_F64);
    let c6 = _mm512_set1_pd(C6_F64);
    let c7 = _mm512_set1_pd(C7_F64);
    let c8 = _mm512_set1_pd(C8_F64);
    let c9 = _mm512_set1_pd(C9_F64);

    // Extract exponent using AVX-512 native 64-bit ops
    let x_bits = _mm512_castpd_si512(x);
    let exp_raw = _mm512_srli_epi64::<52>(x_bits);
    let exp_unbiased = _mm512_sub_epi64(exp_raw, _mm512_set1_epi64(EXP_BIAS_F64));
    let mut n = _mm512_cvtepi64_pd(exp_unbiased);

    // Extract mantissa in [1, 2)
    let mantissa_mask = _mm512_set1_epi64(MANTISSA_MASK_F64 as i64);
    let exp_zero = _mm512_set1_epi64(EXP_ZERO_F64 as i64);
    let m_bits = _mm512_or_si512(_mm512_and_si512(x_bits, mantissa_mask), exp_zero);
    let mut m = _mm512_castsi512_pd(m_bits);

    // Normalize: if m > sqrt(2), divide by 2 and increment exponent
    let need_adjust = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(m, sqrt2);
    m = _mm512_mask_mul_pd(m, need_adjust, m, half);
    n = _mm512_mask_add_pd(n, need_adjust, n, one);

    let f = _mm512_sub_pd(m, one);

    // Horner's method
    let mut poly = c9;
    poly = _mm512_fmadd_pd(poly, f, c8);
    poly = _mm512_fmadd_pd(poly, f, c7);
    poly = _mm512_fmadd_pd(poly, f, c6);
    poly = _mm512_fmadd_pd(poly, f, c5);
    poly = _mm512_fmadd_pd(poly, f, c4);
    poly = _mm512_fmadd_pd(poly, f, c3);
    poly = _mm512_fmadd_pd(poly, f, c2);
    poly = _mm512_fmadd_pd(poly, f, c1);
    poly = _mm512_mul_pd(poly, f);

    _mm512_fmadd_pd(n, ln2, poly)
}

// ============================================================================
// Trigonometric functions: sin, cos, tan
// ============================================================================

/// Fast SIMD sin approximation for f32 using AVX-512
///
/// See `common::_TRIG_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX-512F CPU feature.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn sin_f32(x: __m512) -> __m512 {
    use trig_coefficients::*;

    let two_over_pi = _mm512_set1_ps(std::f32::consts::FRAC_2_PI);
    let pi_over_2 = _mm512_set1_ps(std::f32::consts::FRAC_PI_2);

    let s1 = _mm512_set1_ps(S1_F32);
    let s3 = _mm512_set1_ps(S3_F32);
    let s5 = _mm512_set1_ps(S5_F32);
    let s7 = _mm512_set1_ps(S7_F32);

    let c0 = _mm512_set1_ps(C0_F32);
    let c2 = _mm512_set1_ps(C2_F32);
    let c4 = _mm512_set1_ps(C4_F32);
    let c6 = _mm512_set1_ps(C6_F32);

    // Range reduction
    let j = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
        _mm512_mul_ps(x, two_over_pi),
    );
    let j_int = _mm512_cvtps_epi32(j);

    let y = _mm512_fnmadd_ps(j, pi_over_2, x);

    let y2 = _mm512_mul_ps(y, y);
    let y3 = _mm512_mul_ps(y2, y);
    let y4 = _mm512_mul_ps(y2, y2);
    let y5 = _mm512_mul_ps(y4, y);
    let y6 = _mm512_mul_ps(y4, y2);
    let y7 = _mm512_mul_ps(y4, y3);

    // sin(y) polynomial
    let sin_y = _mm512_fmadd_ps(
        s7,
        y7,
        _mm512_fmadd_ps(s5, y5, _mm512_fmadd_ps(s3, y3, _mm512_mul_ps(s1, y))),
    );

    // cos(y) polynomial
    let cos_y = _mm512_fmadd_ps(c6, y6, _mm512_fmadd_ps(c4, y4, _mm512_fmadd_ps(c2, y2, c0)));

    // Select sin or cos based on quadrant using AVX-512 masks
    let j_mod_4 = _mm512_and_si512(j_int, _mm512_set1_epi32(3));

    // Use cos when j mod 4 is 1 or 3
    let use_cos_mask = _mm512_cmpeq_epi32_mask(
        _mm512_and_si512(j_mod_4, _mm512_set1_epi32(1)),
        _mm512_set1_epi32(1),
    );

    // Negate when j mod 4 is 2 or 3
    let negate_mask = _mm512_cmpeq_epi32_mask(
        _mm512_and_si512(j_mod_4, _mm512_set1_epi32(2)),
        _mm512_set1_epi32(2),
    );

    let result = _mm512_mask_blend_ps(use_cos_mask, sin_y, cos_y);
    let negated = _mm512_sub_ps(_mm512_setzero_ps(), result);
    _mm512_mask_blend_ps(negate_mask, result, negated)
}

/// Fast SIMD sin approximation for f64 using AVX-512
///
/// See `common::_TRIG_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX-512F CPU feature.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn sin_f64(x: __m512d) -> __m512d {
    use trig_coefficients::*;

    let two_over_pi = _mm512_set1_pd(std::f64::consts::FRAC_2_PI);
    let pi_over_2 = _mm512_set1_pd(std::f64::consts::FRAC_PI_2);

    let s1 = _mm512_set1_pd(S1_F64);
    let s3 = _mm512_set1_pd(S3_F64);
    let s5 = _mm512_set1_pd(S5_F64);
    let s7 = _mm512_set1_pd(S7_F64);
    let s9 = _mm512_set1_pd(S9_F64);

    let c0 = _mm512_set1_pd(C0_F64);
    let c2 = _mm512_set1_pd(C2_F64);
    let c4 = _mm512_set1_pd(C4_F64);
    let c6 = _mm512_set1_pd(C6_F64);
    let c8 = _mm512_set1_pd(C8_F64);

    let j = _mm512_roundscale_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
        _mm512_mul_pd(x, two_over_pi),
    );
    let j_i64 = _mm512_cvtpd_epi64(j);

    let y = _mm512_fnmadd_pd(j, pi_over_2, x);

    let y2 = _mm512_mul_pd(y, y);
    let y3 = _mm512_mul_pd(y2, y);
    let y4 = _mm512_mul_pd(y2, y2);
    let y5 = _mm512_mul_pd(y4, y);
    let y6 = _mm512_mul_pd(y4, y2);
    let y7 = _mm512_mul_pd(y4, y3);
    let y8 = _mm512_mul_pd(y4, y4);
    let y9 = _mm512_mul_pd(y8, y);

    // sin(y) polynomial
    let mut sin_y = _mm512_mul_pd(s1, y);
    sin_y = _mm512_fmadd_pd(s3, y3, sin_y);
    sin_y = _mm512_fmadd_pd(s5, y5, sin_y);
    sin_y = _mm512_fmadd_pd(s7, y7, sin_y);
    sin_y = _mm512_fmadd_pd(s9, y9, sin_y);

    // cos(y) polynomial
    let mut cos_y = c0;
    cos_y = _mm512_fmadd_pd(c2, y2, cos_y);
    cos_y = _mm512_fmadd_pd(c4, y4, cos_y);
    cos_y = _mm512_fmadd_pd(c6, y6, cos_y);
    cos_y = _mm512_fmadd_pd(c8, y8, cos_y);

    // Quadrant selection using AVX-512 masks
    let j_mod_4 = _mm512_and_si512(j_i64, _mm512_set1_epi64(3));

    let use_cos_mask = _mm512_cmpeq_epi64_mask(
        _mm512_and_si512(j_mod_4, _mm512_set1_epi64(1)),
        _mm512_set1_epi64(1),
    );

    let negate_mask = _mm512_cmpeq_epi64_mask(
        _mm512_and_si512(j_mod_4, _mm512_set1_epi64(2)),
        _mm512_set1_epi64(2),
    );

    let result = _mm512_mask_blend_pd(use_cos_mask, sin_y, cos_y);
    let negated = _mm512_sub_pd(_mm512_setzero_pd(), result);
    _mm512_mask_blend_pd(negate_mask, result, negated)
}

/// Fast SIMD cos approximation for f32 using AVX-512
///
/// Implemented as: cos(x) = sin(x + π/2)
///
/// # Safety
/// Requires AVX-512F CPU feature.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn cos_f32(x: __m512) -> __m512 {
    let pi_over_2 = _mm512_set1_ps(std::f32::consts::FRAC_PI_2);
    sin_f32(_mm512_add_ps(x, pi_over_2))
}

/// Fast SIMD cos approximation for f64 using AVX-512
///
/// # Safety
/// Requires AVX-512F CPU feature.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn cos_f64(x: __m512d) -> __m512d {
    let pi_over_2 = _mm512_set1_pd(std::f64::consts::FRAC_PI_2);
    sin_f64(_mm512_add_pd(x, pi_over_2))
}

/// Fast SIMD tan approximation for f32 using AVX-512
///
/// See `common::_TAN_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX-512F CPU feature.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn tan_f32(x: __m512) -> __m512 {
    use tan_coefficients::*;

    let two_over_pi = _mm512_set1_ps(std::f32::consts::FRAC_2_PI);
    let pi_over_2 = _mm512_set1_ps(std::f32::consts::FRAC_PI_2);

    let j = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
        _mm512_mul_ps(x, two_over_pi),
    );
    let y = _mm512_fnmadd_ps(j, pi_over_2, x);

    let t1 = _mm512_set1_ps(T1_F32);
    let t3 = _mm512_set1_ps(T3_F32);
    let t5 = _mm512_set1_ps(T5_F32);
    let t7 = _mm512_set1_ps(T7_F32);
    let t9 = _mm512_set1_ps(T9_F32);
    let t11 = _mm512_set1_ps(T11_F32);

    let y2 = _mm512_mul_ps(y, y);

    // Horner's method
    let mut poly = t11;
    poly = _mm512_fmadd_ps(poly, y2, t9);
    poly = _mm512_fmadd_ps(poly, y2, t7);
    poly = _mm512_fmadd_ps(poly, y2, t5);
    poly = _mm512_fmadd_ps(poly, y2, t3);
    poly = _mm512_fmadd_ps(poly, y2, t1);
    let tan_y = _mm512_mul_ps(y, poly);

    // For odd quadrants, use -cot(y)
    let j_int = _mm512_cvtps_epi32(j);
    let use_cot_mask = _mm512_cmpeq_epi32_mask(
        _mm512_and_si512(j_int, _mm512_set1_epi32(1)),
        _mm512_set1_epi32(1),
    );

    let neg_one = _mm512_set1_ps(-1.0);
    let cot_y = _mm512_div_ps(neg_one, tan_y);

    _mm512_mask_blend_ps(use_cot_mask, tan_y, cot_y)
}

/// Fast SIMD tan approximation for f64 using AVX-512
///
/// See `common::_TAN_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX-512F CPU feature.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn tan_f64(x: __m512d) -> __m512d {
    use tan_coefficients::*;

    let two_over_pi = _mm512_set1_pd(std::f64::consts::FRAC_2_PI);
    let pi_over_2 = _mm512_set1_pd(std::f64::consts::FRAC_PI_2);

    let j = _mm512_roundscale_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
        _mm512_mul_pd(x, two_over_pi),
    );
    let y = _mm512_fnmadd_pd(j, pi_over_2, x);

    let t1 = _mm512_set1_pd(T1_F64);
    let t3 = _mm512_set1_pd(T3_F64);
    let t5 = _mm512_set1_pd(T5_F64);
    let t7 = _mm512_set1_pd(T7_F64);
    let t9 = _mm512_set1_pd(T9_F64);
    let t11 = _mm512_set1_pd(T11_F64);
    let t13 = _mm512_set1_pd(T13_F64);

    let y2 = _mm512_mul_pd(y, y);

    // Horner's method
    let mut poly = t13;
    poly = _mm512_fmadd_pd(poly, y2, t11);
    poly = _mm512_fmadd_pd(poly, y2, t9);
    poly = _mm512_fmadd_pd(poly, y2, t7);
    poly = _mm512_fmadd_pd(poly, y2, t5);
    poly = _mm512_fmadd_pd(poly, y2, t3);
    poly = _mm512_fmadd_pd(poly, y2, t1);
    let tan_y = _mm512_mul_pd(y, poly);

    let j_i64 = _mm512_cvtpd_epi64(j);
    let use_cot_mask = _mm512_cmpeq_epi64_mask(
        _mm512_and_si512(j_i64, _mm512_set1_epi64(1)),
        _mm512_set1_epi64(1),
    );

    let neg_one = _mm512_set1_pd(-1.0);
    let cot_y = _mm512_div_pd(neg_one, tan_y);

    _mm512_mask_blend_pd(use_cot_mask, tan_y, cot_y)
}
