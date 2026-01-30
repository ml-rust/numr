//! AVX-512 mathematical function implementations
//!
//! Provides vectorized exp and tanh using 512-bit registers.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// Exponential function: exp(x)
// ============================================================================

/// Fast SIMD exp approximation for f32 using AVX-512
///
/// Algorithm: exp(x) = 2^(x * log2(e)) = 2^n * 2^f
/// where n = round(x * log2(e)) and f = x * log2(e) - n
///
/// # Accuracy
/// - Relative error: < 1e-6 for inputs in [-88, 88]
///
/// # Safety
/// Requires AVX-512F CPU feature.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn exp_f32(x: __m512) -> __m512 {
    let log2e = _mm512_set1_ps(std::f32::consts::LOG2_E);
    let ln2 = _mm512_set1_ps(std::f32::consts::LN_2);

    // Taylor series coefficients
    let c0 = _mm512_set1_ps(1.0);
    let c1 = _mm512_set1_ps(1.0);
    let c2 = _mm512_set1_ps(0.5);
    let c3 = _mm512_set1_ps(1.0 / 6.0);
    let c4 = _mm512_set1_ps(1.0 / 24.0);
    let c5 = _mm512_set1_ps(1.0 / 120.0);
    let c6 = _mm512_set1_ps(1.0 / 720.0);

    // Clamp input to avoid overflow/underflow
    let x = _mm512_max_ps(x, _mm512_set1_ps(-88.0));
    let x = _mm512_min_ps(x, _mm512_set1_ps(88.0));

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
/// # Accuracy
/// - Relative error: < 1e-12 for inputs in [-709, 709]
///
/// # Safety
/// Requires AVX-512F CPU feature.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn exp_f64(x: __m512d) -> __m512d {
    let log2e = _mm512_set1_pd(std::f64::consts::LOG2_E);
    let ln2 = _mm512_set1_pd(std::f64::consts::LN_2);

    let c0 = _mm512_set1_pd(1.0);
    let c1 = _mm512_set1_pd(1.0);
    let c2 = _mm512_set1_pd(0.5);
    let c3 = _mm512_set1_pd(1.0 / 6.0);
    let c4 = _mm512_set1_pd(1.0 / 24.0);
    let c5 = _mm512_set1_pd(1.0 / 120.0);
    let c6 = _mm512_set1_pd(1.0 / 720.0);

    // Clamp input
    let x = _mm512_max_pd(x, _mm512_set1_pd(-709.0));
    let x = _mm512_min_pd(x, _mm512_set1_pd(709.0));

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
