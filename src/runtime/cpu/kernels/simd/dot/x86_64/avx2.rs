//! AVX2 i8 dot product kernels
//!
//! Uses i8 → i16 widening + _mm256_madd_epi16 for correct signed i8 x i8 → i32 accumulation.
//! Processes 32 elements per iteration (two 16-element halves widened to i16).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const I8_LANES: usize = 32; // Process 32 i8s per iteration

/// Horizontal sum of 8 i32 lanes in __m256i
#[target_feature(enable = "avx2")]
unsafe fn hsum_epi32(v: __m256i) -> i32 {
    let hi128 = _mm256_extracti128_si256(v, 1);
    let lo128 = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo128, hi128);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 0b_00_00_00_01);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32)
}

/// Dot product of signed i8 vectors, accumulated in i32.
///
/// Strategy: Load 32 bytes, split into low/high 16 bytes, sign-extend to i16,
/// use _mm256_madd_epi16 (signed i16 pairs → i32) to accumulate.
///
/// # Safety
/// - CPU must support AVX2
/// - Pointers must be valid for `len` elements
#[target_feature(enable = "avx2")]
pub unsafe fn i8xi8_dot_i32(a: *const i8, b: *const i8, len: usize) -> i32 {
    let chunks = len / I8_LANES;
    let remainder = len % I8_LANES;

    let mut acc = _mm256_setzero_si256();

    for i in 0..chunks {
        let offset = i * I8_LANES;
        let va = _mm256_loadu_si256(a.add(offset) as *const __m256i);
        let vb = _mm256_loadu_si256(b.add(offset) as *const __m256i);

        // Process low 16 bytes: sign-extend i8 → i16
        let va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        let vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        // madd: multiply pairs of i16, sum adjacent → i32
        let prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
        acc = _mm256_add_epi32(acc, prod_lo);

        // Process high 16 bytes
        let va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        let vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
        let prod_hi = _mm256_madd_epi16(va_hi, vb_hi);
        acc = _mm256_add_epi32(acc, prod_hi);
    }

    let mut result = hsum_epi32(acc);

    // Scalar tail
    for i in 0..remainder {
        let offset = chunks * I8_LANES + i;
        result += (*a.add(offset) as i32) * (*b.add(offset) as i32);
    }

    result
}

/// Scaled dot product of signed i8 vectors, returning f32.
///
/// Computes scale * sum(a[i] * b[i]) for i in 0..len.
///
/// # Safety
/// - CPU must support AVX2
/// - Pointers must be valid for `len` elements
#[target_feature(enable = "avx2")]
pub unsafe fn i8xi8_dot_f32(a: *const i8, b: *const i8, scale: f32, len: usize) -> f32 {
    (i8xi8_dot_i32(a, b, len) as f32) * scale
}
