//! AVX2 mathematical function implementations
//!
//! Provides vectorized transcendental functions using 256-bit registers.
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
//! | atan     | ✓   | ✓   | < 1e-6 / 1e-12 |
//!
//! # Safety
//!
//! All functions require AVX2 and FMA CPU features.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::common::{
    atan_coefficients, exp_coefficients, log_coefficients, tan_coefficients, trig_coefficients,
};

// ============================================================================
// Exponential function: exp(x)
// ============================================================================

/// Fast SIMD exp approximation for f32 using AVX2+FMA
///
/// See `common::_EXP_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn exp_f32(x: __m256) -> __m256 {
    use exp_coefficients::*;

    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);

    let c0 = _mm256_set1_ps(C0_F32);
    let c1 = _mm256_set1_ps(C1_F32);
    let c2 = _mm256_set1_ps(C2_F32);
    let c3 = _mm256_set1_ps(C3_F32);
    let c4 = _mm256_set1_ps(C4_F32);
    let c5 = _mm256_set1_ps(C5_F32);
    let c6 = _mm256_set1_ps(C6_F32);

    // Clamp input to avoid overflow/underflow
    let x = _mm256_max_ps(x, _mm256_set1_ps(MIN_F32));
    let x = _mm256_min_ps(x, _mm256_set1_ps(MAX_F32));

    // y = x * log2(e)
    let y = _mm256_mul_ps(x, log2e);

    // n = round(y) - integer part
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(y);

    // f = y - n - fractional part in [-0.5, 0.5]
    let f = _mm256_sub_ps(y, n);

    // r = f * ln(2) - convert back to natural log scale
    let r = _mm256_mul_ps(f, ln2);

    // Polynomial approximation using Horner's method
    let r2 = _mm256_mul_ps(r, r);
    let r3 = _mm256_mul_ps(r2, r);
    let r4 = _mm256_mul_ps(r2, r2);
    let r5 = _mm256_mul_ps(r4, r);
    let r6 = _mm256_mul_ps(r4, r2);

    let mut poly = c0;
    poly = _mm256_fmadd_ps(c1, r, poly);
    poly = _mm256_fmadd_ps(c2, r2, poly);
    poly = _mm256_fmadd_ps(c3, r3, poly);
    poly = _mm256_fmadd_ps(c4, r4, poly);
    poly = _mm256_fmadd_ps(c5, r5, poly);
    poly = _mm256_fmadd_ps(c6, r6, poly);

    // Compute 2^n using IEEE 754 bit manipulation
    // 2^n = reinterpret((n + 127) << 23) for f32
    let n_i32 = _mm256_cvtps_epi32(n);
    let bias = _mm256_set1_epi32(127);
    let exp_bits = _mm256_slli_epi32::<23>(_mm256_add_epi32(n_i32, bias));
    let pow2n = _mm256_castsi256_ps(exp_bits);

    // Result = 2^n * exp(r)
    _mm256_mul_ps(pow2n, poly)
}

/// Fast SIMD exp approximation for f64 using AVX2+FMA
///
/// See `common::_EXP_ALGORITHM_DOC` for algorithm details.
///
/// # Note
/// AVX2 lacks native 64-bit integer <-> double conversion. This implementation
/// uses scalar extraction for the 2^n computation, which is the standard
/// workaround. The polynomial computation remains fully vectorized.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn exp_f64(x: __m256d) -> __m256d {
    use exp_coefficients::*;

    let log2e = _mm256_set1_pd(std::f64::consts::LOG2_E);
    let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);

    let c0 = _mm256_set1_pd(C0_F64);
    let c1 = _mm256_set1_pd(C1_F64);
    let c2 = _mm256_set1_pd(C2_F64);
    let c3 = _mm256_set1_pd(C3_F64);
    let c4 = _mm256_set1_pd(C4_F64);
    let c5 = _mm256_set1_pd(C5_F64);
    let c6 = _mm256_set1_pd(C6_F64);

    // Clamp input
    let x = _mm256_max_pd(x, _mm256_set1_pd(MIN_F64));
    let x = _mm256_min_pd(x, _mm256_set1_pd(MAX_F64));

    let y = _mm256_mul_pd(x, log2e);
    let n = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(y);
    let f = _mm256_sub_pd(y, n);
    let r = _mm256_mul_pd(f, ln2);

    let r2 = _mm256_mul_pd(r, r);
    let r3 = _mm256_mul_pd(r2, r);
    let r4 = _mm256_mul_pd(r2, r2);
    let r5 = _mm256_mul_pd(r4, r);
    let r6 = _mm256_mul_pd(r4, r2);

    let mut poly = c0;
    poly = _mm256_fmadd_pd(c1, r, poly);
    poly = _mm256_fmadd_pd(c2, r2, poly);
    poly = _mm256_fmadd_pd(c3, r3, poly);
    poly = _mm256_fmadd_pd(c4, r4, poly);
    poly = _mm256_fmadd_pd(c5, r5, poly);
    poly = _mm256_fmadd_pd(c6, r6, poly);

    // AVX2 lacks _mm256_cvtpd_epi64, use scalar conversion for 2^n
    // This is a known AVX2 limitation - polynomial eval is still SIMD
    let mut result = [0.0f64; 4];
    let mut n_arr = [0.0f64; 4];
    let mut poly_arr = [0.0f64; 4];

    _mm256_storeu_pd(n_arr.as_mut_ptr(), n);
    _mm256_storeu_pd(poly_arr.as_mut_ptr(), poly);

    for i in 0..4 {
        let n_i = n_arr[i] as i64;
        let exp_bits = ((n_i + 1023) as u64) << 52;
        let pow2n = f64::from_bits(exp_bits);
        result[i] = pow2n * poly_arr[i];
    }

    _mm256_loadu_pd(result.as_ptr())
}

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
// Natural logarithm: log(x)
// ============================================================================

/// Fast SIMD log approximation for f32 using AVX2+FMA
///
/// See `common::_LOG_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log_f32(x: __m256) -> __m256 {
    use log_coefficients::*;

    let one = _mm256_set1_ps(1.0);
    let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
    let sqrt2 = _mm256_set1_ps(std::f32::consts::SQRT_2);
    let half = _mm256_set1_ps(0.5);

    let c1 = _mm256_set1_ps(C1_F32);
    let c2 = _mm256_set1_ps(C2_F32);
    let c3 = _mm256_set1_ps(C3_F32);
    let c4 = _mm256_set1_ps(C4_F32);
    let c5 = _mm256_set1_ps(C5_F32);
    let c6 = _mm256_set1_ps(C6_F32);
    let c7 = _mm256_set1_ps(C7_F32);

    // Extract exponent: reinterpret as int, shift right by 23, subtract bias
    let x_bits = _mm256_castps_si256(x);
    let exp_raw = _mm256_srli_epi32::<23>(x_bits);
    let exp_unbiased = _mm256_sub_epi32(exp_raw, _mm256_set1_epi32(EXP_BIAS_F32));
    let mut n = _mm256_cvtepi32_ps(exp_unbiased);

    // Extract mantissa and set exponent to 0 (so mantissa is in [1, 2))
    let mantissa_mask = _mm256_set1_epi32(MANTISSA_MASK_F32);
    let exp_zero = _mm256_set1_epi32(EXP_ZERO_F32);
    let m_bits = _mm256_or_si256(_mm256_and_si256(x_bits, mantissa_mask), exp_zero);
    let mut m = _mm256_castsi256_ps(m_bits);

    // Normalize: if m > sqrt(2), divide by 2 and increment exponent
    // This keeps f in [-0.2929, 0.4142] for better polynomial accuracy
    let need_adjust = _mm256_cmp_ps::<_CMP_GT_OQ>(m, sqrt2);
    m = _mm256_blendv_ps(m, _mm256_mul_ps(m, half), need_adjust);
    n = _mm256_blendv_ps(n, _mm256_add_ps(n, one), need_adjust);

    // f = m - 1, so log(m) = log(1 + f), f is now in [-0.2929, 0.4142]
    let f = _mm256_sub_ps(m, one);

    // Horner's method: ((((((c7*f + c6)*f + c5)*f + c4)*f + c3)*f + c2)*f + c1)*f
    let mut poly = c7;
    poly = _mm256_fmadd_ps(poly, f, c6);
    poly = _mm256_fmadd_ps(poly, f, c5);
    poly = _mm256_fmadd_ps(poly, f, c4);
    poly = _mm256_fmadd_ps(poly, f, c3);
    poly = _mm256_fmadd_ps(poly, f, c2);
    poly = _mm256_fmadd_ps(poly, f, c1);
    poly = _mm256_mul_ps(poly, f);

    // Result = n * ln(2) + log(m)
    _mm256_fmadd_ps(n, ln2, poly)
}

/// Fast SIMD log approximation for f64 using AVX2+FMA
///
/// See `common::_LOG_ALGORITHM_DOC` for algorithm details.
///
/// # Implementation Note
/// Unlike the naive scalar-loop approach, this implementation uses native AVX2
/// 64-bit SIMD operations for exponent extraction. The only scalar operations
/// are for the normalization conditional and final reconstruction, which cannot
/// be avoided due to AVX2's lack of 64-bit comparison and conversion intrinsics.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log_f64(x: __m256d) -> __m256d {
    use log_coefficients::*;

    let one = _mm256_set1_pd(1.0);
    let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
    let sqrt2_val = std::f64::consts::SQRT_2;

    let c1 = _mm256_set1_pd(C1_F64);
    let c2 = _mm256_set1_pd(C2_F64);
    let c3 = _mm256_set1_pd(C3_F64);
    let c4 = _mm256_set1_pd(C4_F64);
    let c5 = _mm256_set1_pd(C5_F64);
    let c6 = _mm256_set1_pd(C6_F64);
    let c7 = _mm256_set1_pd(C7_F64);
    let c8 = _mm256_set1_pd(C8_F64);
    let c9 = _mm256_set1_pd(C9_F64);

    // Use SIMD for bit manipulation - AVX2 has 64-bit shifts
    let x_bits = _mm256_castpd_si256(x);

    // Extract exponent using 64-bit SIMD shift
    let exp_raw = _mm256_srli_epi64::<52>(x_bits);

    // Extract mantissa and set exponent to bias (so mantissa is in [1, 2))
    let mantissa_mask = _mm256_set1_epi64x(MANTISSA_MASK_F64 as i64);
    let exp_zero = _mm256_set1_epi64x(EXP_ZERO_F64 as i64);
    let m_bits = _mm256_or_si256(_mm256_and_si256(x_bits, mantissa_mask), exp_zero);
    let m_initial = _mm256_castsi256_pd(m_bits);

    // AVX2 lacks 64-bit int comparison and conversion, so we extract for
    // normalization and exponent calculation. The heavy lifting (polynomial
    // evaluation) remains fully vectorized.
    let mut m_arr = [0.0f64; 4];
    let mut exp_arr = [0i64; 4];
    _mm256_storeu_pd(m_arr.as_mut_ptr(), m_initial);
    _mm256_storeu_si256(exp_arr.as_mut_ptr() as *mut __m256i, exp_raw);

    let mut n_arr = [0.0f64; 4];
    for i in 0..4 {
        let mut exp_unbiased = exp_arr[i] - EXP_BIAS_F64;
        let mut m = m_arr[i];

        // Normalize: if m > sqrt(2), divide by 2 and increment exponent
        if m > sqrt2_val {
            m *= 0.5;
            exp_unbiased += 1;
        }

        n_arr[i] = exp_unbiased as f64;
        m_arr[i] = m;
    }

    let n = _mm256_loadu_pd(n_arr.as_ptr());
    let m = _mm256_loadu_pd(m_arr.as_ptr());

    // f = m - 1 (fully SIMD from here)
    let f = _mm256_sub_pd(m, one);

    // Horner's method for polynomial (fully vectorized)
    let mut poly = c9;
    poly = _mm256_fmadd_pd(poly, f, c8);
    poly = _mm256_fmadd_pd(poly, f, c7);
    poly = _mm256_fmadd_pd(poly, f, c6);
    poly = _mm256_fmadd_pd(poly, f, c5);
    poly = _mm256_fmadd_pd(poly, f, c4);
    poly = _mm256_fmadd_pd(poly, f, c3);
    poly = _mm256_fmadd_pd(poly, f, c2);
    poly = _mm256_fmadd_pd(poly, f, c1);
    poly = _mm256_mul_pd(poly, f);

    // Result = n * ln(2) + log(m) (fully SIMD)
    _mm256_fmadd_pd(n, ln2, poly)
}

// ============================================================================
// Trigonometric functions: sin, cos, tan
// ============================================================================

/// Fast SIMD sin approximation for f32 using AVX2+FMA
///
/// See `common::_TRIG_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sin_f32(x: __m256) -> __m256 {
    use trig_coefficients::*;

    let two_over_pi = _mm256_set1_ps(std::f32::consts::FRAC_2_PI);
    let pi_over_2 = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);

    let s1 = _mm256_set1_ps(S1_F32);
    let s3 = _mm256_set1_ps(S3_F32);
    let s5 = _mm256_set1_ps(S5_F32);
    let s7 = _mm256_set1_ps(S7_F32);

    let c0 = _mm256_set1_ps(C0_F32);
    let c2 = _mm256_set1_ps(C2_F32);
    let c4 = _mm256_set1_ps(C4_F32);
    let c6 = _mm256_set1_ps(C6_F32);

    // Range reduction: j = round(x * 2/π), y = x - j * π/2
    let j = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        two_over_pi,
    ));
    let j_int = _mm256_cvtps_epi32(j);

    let y = _mm256_fnmadd_ps(j, pi_over_2, x);

    let y2 = _mm256_mul_ps(y, y);
    let y3 = _mm256_mul_ps(y2, y);
    let y4 = _mm256_mul_ps(y2, y2);
    let y5 = _mm256_mul_ps(y4, y);
    let y6 = _mm256_mul_ps(y4, y2);
    let y7 = _mm256_mul_ps(y4, y3);

    // sin(y) polynomial
    let sin_y = _mm256_fmadd_ps(
        s7,
        y7,
        _mm256_fmadd_ps(s5, y5, _mm256_fmadd_ps(s3, y3, _mm256_mul_ps(s1, y))),
    );

    // cos(y) polynomial
    let cos_y = _mm256_fmadd_ps(c6, y6, _mm256_fmadd_ps(c4, y4, _mm256_fmadd_ps(c2, y2, c0)));

    // Select sin or cos based on j mod 4
    // j mod 4 = 0: sin(y), 1: cos(y), 2: -sin(y), 3: -cos(y)
    let j_mod_4 = _mm256_and_si256(j_int, _mm256_set1_epi32(3));

    // Use cos when j mod 4 is 1 or 3
    let use_cos_mask = _mm256_cmpeq_epi32(
        _mm256_and_si256(j_mod_4, _mm256_set1_epi32(1)),
        _mm256_set1_epi32(1),
    );
    let use_cos_mask = _mm256_castsi256_ps(use_cos_mask);

    // Negate when j mod 4 is 2 or 3
    let negate_mask = _mm256_cmpeq_epi32(
        _mm256_and_si256(j_mod_4, _mm256_set1_epi32(2)),
        _mm256_set1_epi32(2),
    );
    let negate_mask = _mm256_castsi256_ps(negate_mask);
    let sign_bit = _mm256_set1_ps(-0.0); // Just the sign bit

    let result = _mm256_blendv_ps(sin_y, cos_y, use_cos_mask);
    let negated = _mm256_xor_ps(result, sign_bit);
    _mm256_blendv_ps(result, negated, negate_mask)
}

/// Fast SIMD sin approximation for f64 using AVX2+FMA
///
/// See `common::_TRIG_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sin_f64(x: __m256d) -> __m256d {
    use trig_coefficients::*;

    let two_over_pi = _mm256_set1_pd(std::f64::consts::FRAC_2_PI);
    let pi_over_2 = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);

    let s1 = _mm256_set1_pd(S1_F64);
    let s3 = _mm256_set1_pd(S3_F64);
    let s5 = _mm256_set1_pd(S5_F64);
    let s7 = _mm256_set1_pd(S7_F64);
    let s9 = _mm256_set1_pd(S9_F64);

    let c0 = _mm256_set1_pd(C0_F64);
    let c2 = _mm256_set1_pd(C2_F64);
    let c4 = _mm256_set1_pd(C4_F64);
    let c6 = _mm256_set1_pd(C6_F64);
    let c8 = _mm256_set1_pd(C8_F64);

    let j = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_pd(
        x,
        two_over_pi,
    ));

    // Get j as integers for quadrant selection (AVX2 lacks 64-bit int conversion)
    let mut j_arr = [0.0f64; 4];
    _mm256_storeu_pd(j_arr.as_mut_ptr(), j);
    let j_int: [i32; 4] = [
        j_arr[0] as i32,
        j_arr[1] as i32,
        j_arr[2] as i32,
        j_arr[3] as i32,
    ];

    let y = _mm256_fnmadd_pd(j, pi_over_2, x);

    let y2 = _mm256_mul_pd(y, y);
    let y3 = _mm256_mul_pd(y2, y);
    let y4 = _mm256_mul_pd(y2, y2);
    let y5 = _mm256_mul_pd(y4, y);
    let y6 = _mm256_mul_pd(y4, y2);
    let y7 = _mm256_mul_pd(y4, y3);
    let y8 = _mm256_mul_pd(y4, y4);
    let y9 = _mm256_mul_pd(y8, y);

    // sin(y) and cos(y) polynomials
    let mut sin_y = _mm256_mul_pd(s1, y);
    sin_y = _mm256_fmadd_pd(s3, y3, sin_y);
    sin_y = _mm256_fmadd_pd(s5, y5, sin_y);
    sin_y = _mm256_fmadd_pd(s7, y7, sin_y);
    sin_y = _mm256_fmadd_pd(s9, y9, sin_y);

    let mut cos_y = c0;
    cos_y = _mm256_fmadd_pd(c2, y2, cos_y);
    cos_y = _mm256_fmadd_pd(c4, y4, cos_y);
    cos_y = _mm256_fmadd_pd(c6, y6, cos_y);
    cos_y = _mm256_fmadd_pd(c8, y8, cos_y);

    // Compute result per-element based on quadrant
    let mut sin_arr = [0.0f64; 4];
    let mut cos_arr = [0.0f64; 4];
    _mm256_storeu_pd(sin_arr.as_mut_ptr(), sin_y);
    _mm256_storeu_pd(cos_arr.as_mut_ptr(), cos_y);

    let mut result = [0.0f64; 4];
    for i in 0..4 {
        let quadrant = j_int[i] & 3;
        result[i] = match quadrant {
            0 => sin_arr[i],
            1 => cos_arr[i],
            2 => -sin_arr[i],
            3 => -cos_arr[i],
            _ => unreachable!(),
        };
    }

    _mm256_loadu_pd(result.as_ptr())
}

/// Fast SIMD cos approximation for f32 using AVX2+FMA
///
/// Implemented as: cos(x) = sin(x + π/2)
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cos_f32(x: __m256) -> __m256 {
    let pi_over_2 = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);
    sin_f32(_mm256_add_ps(x, pi_over_2))
}

/// Fast SIMD cos approximation for f64 using AVX2+FMA
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cos_f64(x: __m256d) -> __m256d {
    let pi_over_2 = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);
    sin_f64(_mm256_add_pd(x, pi_over_2))
}

/// Fast SIMD tan approximation for f32 using AVX2+FMA
///
/// See `common::_TAN_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn tan_f32(x: __m256) -> __m256 {
    use tan_coefficients::*;

    let two_over_pi = _mm256_set1_ps(std::f32::consts::FRAC_2_PI);
    let pi_over_2 = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);

    // Range reduction
    let j = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        two_over_pi,
    ));
    let y = _mm256_fnmadd_ps(j, pi_over_2, x);

    let t1 = _mm256_set1_ps(T1_F32);
    let t3 = _mm256_set1_ps(T3_F32);
    let t5 = _mm256_set1_ps(T5_F32);
    let t7 = _mm256_set1_ps(T7_F32);
    let t9 = _mm256_set1_ps(T9_F32);
    let t11 = _mm256_set1_ps(T11_F32);

    let y2 = _mm256_mul_ps(y, y);

    // Horner's method: tan(y) ≈ y * (1 + y²*(t3 + y²*(t5 + y²*(t7 + y²*(t9 + y²*t11)))))
    let mut poly = t11;
    poly = _mm256_fmadd_ps(poly, y2, t9);
    poly = _mm256_fmadd_ps(poly, y2, t7);
    poly = _mm256_fmadd_ps(poly, y2, t5);
    poly = _mm256_fmadd_ps(poly, y2, t3);
    poly = _mm256_fmadd_ps(poly, y2, t1);
    let tan_y = _mm256_mul_ps(y, poly);

    // For quadrants 1 and 3, tan(y + π/2) = -1/tan(y) = -cot(y)
    let j_int = _mm256_cvtps_epi32(j);
    let use_cot_mask = _mm256_cmpeq_epi32(
        _mm256_and_si256(j_int, _mm256_set1_epi32(1)),
        _mm256_set1_epi32(1),
    );
    let use_cot_mask = _mm256_castsi256_ps(use_cot_mask);

    let neg_one = _mm256_set1_ps(-1.0);
    let cot_y = _mm256_div_ps(neg_one, tan_y);

    _mm256_blendv_ps(tan_y, cot_y, use_cot_mask)
}

/// Fast SIMD tan approximation for f64 using AVX2+FMA
///
/// See `common::_TAN_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn tan_f64(x: __m256d) -> __m256d {
    use tan_coefficients::*;

    let two_over_pi = _mm256_set1_pd(std::f64::consts::FRAC_2_PI);
    let pi_over_2 = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);

    let j = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_pd(
        x,
        two_over_pi,
    ));
    let y = _mm256_fnmadd_pd(j, pi_over_2, x);

    let t1 = _mm256_set1_pd(T1_F64);
    let t3 = _mm256_set1_pd(T3_F64);
    let t5 = _mm256_set1_pd(T5_F64);
    let t7 = _mm256_set1_pd(T7_F64);
    let t9 = _mm256_set1_pd(T9_F64);
    let t11 = _mm256_set1_pd(T11_F64);
    let t13 = _mm256_set1_pd(T13_F64);

    let y2 = _mm256_mul_pd(y, y);

    // Horner's method
    let mut poly = t13;
    poly = _mm256_fmadd_pd(poly, y2, t11);
    poly = _mm256_fmadd_pd(poly, y2, t9);
    poly = _mm256_fmadd_pd(poly, y2, t7);
    poly = _mm256_fmadd_pd(poly, y2, t5);
    poly = _mm256_fmadd_pd(poly, y2, t3);
    poly = _mm256_fmadd_pd(poly, y2, t1);
    let tan_y = _mm256_mul_pd(y, poly);

    // Handle quadrant for cotangent (AVX2 lacks 64-bit int comparison)
    let mut j_arr = [0.0f64; 4];
    let mut tan_arr = [0.0f64; 4];
    _mm256_storeu_pd(j_arr.as_mut_ptr(), j);
    _mm256_storeu_pd(tan_arr.as_mut_ptr(), tan_y);

    let mut result = [0.0f64; 4];
    for i in 0..4 {
        let j_int = j_arr[i] as i32;
        result[i] = if (j_int & 1) == 1 {
            -1.0 / tan_arr[i]
        } else {
            tan_arr[i]
        };
    }

    _mm256_loadu_pd(result.as_ptr())
}

// ============================================================================
// Inverse tangent function: atan(x)
// ============================================================================

/// Fast SIMD atan approximation for f32 using AVX2+FMA
///
/// See `common::_ATAN_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn atan_f32(x: __m256) -> __m256 {
    use atan_coefficients::*;

    let one = _mm256_set1_ps(1.0);
    let pi_over_2 = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);

    // Save sign and work with absolute value
    let sign_mask = _mm256_set1_ps(-0.0); // 0x80000000
    let sign = _mm256_and_ps(x, sign_mask);
    let abs_x = _mm256_andnot_ps(sign_mask, x);

    // Range reduction: for |x| > 1, compute atan(1/x) then adjust
    let need_recip = _mm256_cmp_ps::<_CMP_GT_OQ>(abs_x, one);
    let recip_x = _mm256_div_ps(one, abs_x);
    let y = _mm256_blendv_ps(abs_x, recip_x, need_recip);

    // Polynomial approximation for atan(y) where y in [0, 1]
    let a0 = _mm256_set1_ps(A0_F32);
    let a2 = _mm256_set1_ps(A2_F32);
    let a4 = _mm256_set1_ps(A4_F32);
    let a6 = _mm256_set1_ps(A6_F32);
    let a8 = _mm256_set1_ps(A8_F32);
    let a10 = _mm256_set1_ps(A10_F32);
    let a12 = _mm256_set1_ps(A12_F32);

    let y2 = _mm256_mul_ps(y, y);

    // Horner's method: a0 + y²*(a2 + y²*(a4 + y²*(a6 + y²*(a8 + y²*(a10 + y²*a12)))))
    let mut poly = a12;
    poly = _mm256_fmadd_ps(poly, y2, a10);
    poly = _mm256_fmadd_ps(poly, y2, a8);
    poly = _mm256_fmadd_ps(poly, y2, a6);
    poly = _mm256_fmadd_ps(poly, y2, a4);
    poly = _mm256_fmadd_ps(poly, y2, a2);
    poly = _mm256_fmadd_ps(poly, y2, a0);
    let atan_y = _mm256_mul_ps(y, poly);

    // Apply range reduction inverse: if |x| > 1, result = π/2 - atan(1/x)
    let adjusted = _mm256_sub_ps(pi_over_2, atan_y);
    let result = _mm256_blendv_ps(atan_y, adjusted, need_recip);

    // Restore sign
    _mm256_or_ps(result, sign)
}

/// Fast SIMD atan approximation for f64 using AVX2+FMA
///
/// See `common::_ATAN_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn atan_f64(x: __m256d) -> __m256d {
    use atan_coefficients::*;

    let one = _mm256_set1_pd(1.0);
    let pi_over_2 = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);

    // Save sign and work with absolute value
    let sign_mask = _mm256_set1_pd(-0.0); // 0x8000000000000000
    let sign = _mm256_and_pd(x, sign_mask);
    let abs_x = _mm256_andnot_pd(sign_mask, x);

    // Range reduction: for |x| > 1, compute atan(1/x) then adjust
    let need_recip = _mm256_cmp_pd::<_CMP_GT_OQ>(abs_x, one);
    let recip_x = _mm256_div_pd(one, abs_x);
    let y = _mm256_blendv_pd(abs_x, recip_x, need_recip);

    // Polynomial approximation for atan(y) where y in [0, 1]
    let a0 = _mm256_set1_pd(A0_F64);
    let a2 = _mm256_set1_pd(A2_F64);
    let a4 = _mm256_set1_pd(A4_F64);
    let a6 = _mm256_set1_pd(A6_F64);
    let a8 = _mm256_set1_pd(A8_F64);
    let a10 = _mm256_set1_pd(A10_F64);
    let a12 = _mm256_set1_pd(A12_F64);
    let a14 = _mm256_set1_pd(A14_F64);
    let a16 = _mm256_set1_pd(A16_F64);
    let a18 = _mm256_set1_pd(A18_F64);
    let a20 = _mm256_set1_pd(A20_F64);

    let y2 = _mm256_mul_pd(y, y);

    // Horner's method with 11 terms for higher precision
    let mut poly = a20;
    poly = _mm256_fmadd_pd(poly, y2, a18);
    poly = _mm256_fmadd_pd(poly, y2, a16);
    poly = _mm256_fmadd_pd(poly, y2, a14);
    poly = _mm256_fmadd_pd(poly, y2, a12);
    poly = _mm256_fmadd_pd(poly, y2, a10);
    poly = _mm256_fmadd_pd(poly, y2, a8);
    poly = _mm256_fmadd_pd(poly, y2, a6);
    poly = _mm256_fmadd_pd(poly, y2, a4);
    poly = _mm256_fmadd_pd(poly, y2, a2);
    poly = _mm256_fmadd_pd(poly, y2, a0);
    let atan_y = _mm256_mul_pd(y, poly);

    // Apply range reduction inverse: if |x| > 1, result = π/2 - atan(1/x)
    let adjusted = _mm256_sub_pd(pi_over_2, atan_y);
    let result = _mm256_blendv_pd(atan_y, adjusted, need_recip);

    // Restore sign
    _mm256_or_pd(result, sign)
}

// ============================================================================
// Horizontal reductions
// ============================================================================

/// Horizontal maximum of 8 f32 values in an AVX2 register
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn hmax_f32(v: __m256) -> f32 {
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let max128 = _mm_max_ps(low, high);
    let shuf = _mm_movehdup_ps(max128);
    let max64 = _mm_max_ps(max128, shuf);
    let shuf2 = _mm_movehl_ps(max64, max64);
    let max32 = _mm_max_ss(max64, shuf2);
    _mm_cvtss_f32(max32)
}

/// Horizontal maximum of 4 f64 values in an AVX2 register
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn hmax_f64(v: __m256d) -> f64 {
    let high = _mm256_extractf128_pd(v, 1);
    let low = _mm256_castpd256_pd128(v);
    let max128 = _mm_max_pd(low, high);
    let shuf = _mm_unpackhi_pd(max128, max128);
    let max64 = _mm_max_sd(max128, shuf);
    _mm_cvtsd_f64(max64)
}

/// Horizontal sum of 8 f32 values in an AVX2 register
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn hsum_f32(v: __m256) -> f32 {
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(low, high);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}

/// Horizontal sum of 4 f64 values in an AVX2 register
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn hsum_f64(v: __m256d) -> f64 {
    let high = _mm256_extractf128_pd(v, 1);
    let low = _mm256_castpd256_pd128(v);
    let sum128 = _mm_add_pd(low, high);
    let shuf = _mm_unpackhi_pd(sum128, sum128);
    let sum64 = _mm_add_sd(sum128, shuf);
    _mm_cvtsd_f64(sum64)
}

// ============================================================================
// Additional transcendental functions
// ============================================================================

/// Fast SIMD rsqrt (1/sqrt(x)) for f32 using AVX2
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
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn rsqrt_f64(x: __m256d) -> __m256d {
    let sqrt_x = _mm256_sqrt_pd(x);
    _mm256_div_pd(_mm256_set1_pd(1.0), sqrt_x)
}

/// Fast SIMD exp2 (2^x) for f32 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn exp2_f32(x: __m256) -> __m256 {
    // 2^x = e^(x * ln(2))
    let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
    exp_f32(_mm256_mul_ps(x, ln2))
}

/// Fast SIMD exp2 (2^x) for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn exp2_f64(x: __m256d) -> __m256d {
    let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
    exp_f64(_mm256_mul_pd(x, ln2))
}

/// Fast SIMD expm1 (e^x - 1) for f32 using AVX2
/// Uses direct computation for |x| > 0.5, Taylor series for small x
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn expm1_f32(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);

    // For small |x|, use Taylor series: x + x^2/2 + x^3/6 + x^4/24
    let x2 = _mm256_mul_ps(x, x);
    let x3 = _mm256_mul_ps(x2, x);
    let x4 = _mm256_mul_ps(x2, x2);
    let c2 = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(1.0 / 6.0);
    let c4 = _mm256_set1_ps(1.0 / 24.0);
    let taylor = _mm256_fmadd_ps(c4, x4, _mm256_fmadd_ps(c3, x3, _mm256_fmadd_ps(c2, x2, x)));

    // For large |x|, use exp(x) - 1
    let exp_result = _mm256_sub_ps(exp_f32(x), one);

    // Blend based on |x| > 0.5
    let mask = _mm256_cmp_ps::<_CMP_GT_OQ>(abs_x, half);
    _mm256_blendv_ps(taylor, exp_result, mask)
}

/// Fast SIMD expm1 (e^x - 1) for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn expm1_f64(x: __m256d) -> __m256d {
    let one = _mm256_set1_pd(1.0);
    let half = _mm256_set1_pd(0.5);
    let abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);

    let x2 = _mm256_mul_pd(x, x);
    let x3 = _mm256_mul_pd(x2, x);
    let x4 = _mm256_mul_pd(x2, x2);
    let c2 = _mm256_set1_pd(0.5);
    let c3 = _mm256_set1_pd(1.0 / 6.0);
    let c4 = _mm256_set1_pd(1.0 / 24.0);
    let taylor = _mm256_fmadd_pd(c4, x4, _mm256_fmadd_pd(c3, x3, _mm256_fmadd_pd(c2, x2, x)));

    let exp_result = _mm256_sub_pd(exp_f64(x), one);
    let mask = _mm256_cmp_pd::<_CMP_GT_OQ>(abs_x, half);
    _mm256_blendv_pd(taylor, exp_result, mask)
}

/// Fast SIMD log2 for f32 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log2_f32(x: __m256) -> __m256 {
    // log2(x) = log(x) * log2(e)
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    _mm256_mul_ps(log_f32(x), log2e)
}

/// Fast SIMD log2 for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log2_f64(x: __m256d) -> __m256d {
    let log2e = _mm256_set1_pd(std::f64::consts::LOG2_E);
    _mm256_mul_pd(log_f64(x), log2e)
}

/// Fast SIMD log10 for f32 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log10_f32(x: __m256) -> __m256 {
    // log10(x) = log(x) * log10(e)
    let log10e = _mm256_set1_ps(std::f32::consts::LOG10_E);
    _mm256_mul_ps(log_f32(x), log10e)
}

/// Fast SIMD log10 for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log10_f64(x: __m256d) -> __m256d {
    let log10e = _mm256_set1_pd(std::f64::consts::LOG10_E);
    _mm256_mul_pd(log_f64(x), log10e)
}

/// Fast SIMD log1p (log(1+x)) for f32 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log1p_f32(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);

    // For small |x|, use Taylor series: x - x^2/2 + x^3/3 - x^4/4
    let x2 = _mm256_mul_ps(x, x);
    let x3 = _mm256_mul_ps(x2, x);
    let x4 = _mm256_mul_ps(x2, x2);
    let c2 = _mm256_set1_ps(-0.5);
    let c3 = _mm256_set1_ps(1.0 / 3.0);
    let c4 = _mm256_set1_ps(-0.25);
    let taylor = _mm256_fmadd_ps(c4, x4, _mm256_fmadd_ps(c3, x3, _mm256_fmadd_ps(c2, x2, x)));

    // For large |x|, use log(1 + x)
    let log_result = log_f32(_mm256_add_ps(one, x));

    let mask = _mm256_cmp_ps::<_CMP_GT_OQ>(abs_x, half);
    _mm256_blendv_ps(taylor, log_result, mask)
}

/// Fast SIMD log1p (log(1+x)) for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log1p_f64(x: __m256d) -> __m256d {
    let one = _mm256_set1_pd(1.0);
    let half = _mm256_set1_pd(0.5);
    let abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);

    let x2 = _mm256_mul_pd(x, x);
    let x3 = _mm256_mul_pd(x2, x);
    let x4 = _mm256_mul_pd(x2, x2);
    let c2 = _mm256_set1_pd(-0.5);
    let c3 = _mm256_set1_pd(1.0 / 3.0);
    let c4 = _mm256_set1_pd(-0.25);
    let taylor = _mm256_fmadd_pd(c4, x4, _mm256_fmadd_pd(c3, x3, _mm256_fmadd_pd(c2, x2, x)));

    let log_result = log_f64(_mm256_add_pd(one, x));
    let mask = _mm256_cmp_pd::<_CMP_GT_OQ>(abs_x, half);
    _mm256_blendv_pd(taylor, log_result, mask)
}

/// Fast SIMD sinh for f32 using AVX2
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
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sinh_f64(x: __m256d) -> __m256d {
    let half = _mm256_set1_pd(0.5);
    let exp_x = exp_f64(x);
    let exp_neg_x = exp_f64(_mm256_sub_pd(_mm256_setzero_pd(), x));
    _mm256_mul_pd(half, _mm256_sub_pd(exp_x, exp_neg_x))
}

/// Fast SIMD cosh for f32 using AVX2
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
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cosh_f64(x: __m256d) -> __m256d {
    let half = _mm256_set1_pd(0.5);
    let exp_x = exp_f64(x);
    let exp_neg_x = exp_f64(_mm256_sub_pd(_mm256_setzero_pd(), x));
    _mm256_mul_pd(half, _mm256_add_pd(exp_x, exp_neg_x))
}

/// Fast SIMD asinh for f32 using AVX2
/// asinh(x) = log(x + sqrt(x^2 + 1))
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn asinh_f32(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let x2 = _mm256_mul_ps(x, x);
    let sqrt_term = _mm256_sqrt_ps(_mm256_add_ps(x2, one));
    log_f32(_mm256_add_ps(x, sqrt_term))
}

/// Fast SIMD asinh for f64 using AVX2
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
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn acosh_f32(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let x2 = _mm256_mul_ps(x, x);
    let sqrt_term = _mm256_sqrt_ps(_mm256_sub_ps(x2, one));
    log_f32(_mm256_add_ps(x, sqrt_term))
}

/// Fast SIMD acosh for f64 using AVX2
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

/// Fast SIMD asin for f32 using AVX2
/// Uses polynomial approximation with range reduction
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn asin_f32(x: __m256) -> __m256 {
    // asin(x) = atan(x / sqrt(1 - x^2))
    let one = _mm256_set1_ps(1.0);
    let x2 = _mm256_mul_ps(x, x);
    let sqrt_term = _mm256_sqrt_ps(_mm256_sub_ps(one, x2));
    let ratio = _mm256_div_ps(x, sqrt_term);
    atan_f32(ratio)
}

/// Fast SIMD asin for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn asin_f64(x: __m256d) -> __m256d {
    let one = _mm256_set1_pd(1.0);
    let x2 = _mm256_mul_pd(x, x);
    let sqrt_term = _mm256_sqrt_pd(_mm256_sub_pd(one, x2));
    let ratio = _mm256_div_pd(x, sqrt_term);
    atan_f64(ratio)
}

/// Fast SIMD acos for f32 using AVX2
/// acos(x) = pi/2 - asin(x)
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn acos_f32(x: __m256) -> __m256 {
    let pi_half = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);
    _mm256_sub_ps(pi_half, asin_f32(x))
}

/// Fast SIMD acos for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn acos_f64(x: __m256d) -> __m256d {
    let pi_half = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);
    _mm256_sub_pd(pi_half, asin_f64(x))
}

/// Fast SIMD cbrt (cube root) for f32 using AVX2
/// Uses Halley's method for refinement
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
