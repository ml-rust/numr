//! AVX2 trigonometric function implementations (sin, cos, tan, atan, asin, acos)
//!
//! # Safety
//!
//! All functions require AVX2 and FMA CPU features.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::common::{atan_coefficients, tan_coefficients, trig_coefficients};

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
// Inverse trigonometric functions: asin, acos
// ============================================================================

/// Fast SIMD asin for f32 using AVX2
/// Uses polynomial approximation with range reduction
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
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
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
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
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn acos_f32(x: __m256) -> __m256 {
    let pi_half = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);
    _mm256_sub_ps(pi_half, asin_f32(x))
}

/// Fast SIMD acos for f64 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn acos_f64(x: __m256d) -> __m256d {
    let pi_half = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);
    _mm256_sub_pd(pi_half, asin_f64(x))
}
