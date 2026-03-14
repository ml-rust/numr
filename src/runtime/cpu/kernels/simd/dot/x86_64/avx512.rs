//! AVX-512 i8 dot product kernels
//!
//! Uses i8 → i16 widening + _mm512_madd_epi16 for correct signed i8 x i8 → i32 accumulation.
//! Processes 64 elements per iteration (two 32-element halves widened to i16).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const I8_LANES: usize = 64; // Process 64 i8s per iteration

/// Horizontal sum of 16 i32 lanes in __m512i
#[target_feature(enable = "avx512f")]
unsafe fn hsum_epi32_512(v: __m512i) -> i32 {
    let lo256 = _mm512_castsi512_si256(v);
    let hi256 = _mm512_extracti64x4_epi64(v, 1);
    let sum256 = _mm256_add_epi32(lo256, hi256);
    let hi128 = _mm256_extracti128_si256(sum256, 1);
    let lo128 = _mm256_castsi256_si128(sum256);
    let sum128 = _mm_add_epi32(lo128, hi128);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 0b_00_00_00_01);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32)
}

/// Dot product of signed i8 vectors using AVX-512BW, accumulated in i32.
///
/// Strategy: Load 64 bytes, split into low/high 32 bytes, sign-extend to i16,
/// use _mm512_madd_epi16 (signed i16 pairs → i32) to accumulate.
///
/// # Safety
/// - CPU must support AVX-512F + AVX-512BW
/// - Pointers must be valid for `len` elements
#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn i8xi8_dot_i32(a: *const i8, b: *const i8, len: usize) -> i32 {
    let chunks = len / I8_LANES;
    let remainder = len % I8_LANES;

    let mut acc = _mm512_setzero_si512();

    for i in 0..chunks {
        let offset = i * I8_LANES;
        let va = _mm512_loadu_si512(a.add(offset) as *const __m512i);
        let vb = _mm512_loadu_si512(b.add(offset) as *const __m512i);

        // Process low 32 bytes: sign-extend i8 → i16 in 512-bit
        let va_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(va));
        let vb_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(vb));
        let prod_lo = _mm512_madd_epi16(va_lo, vb_lo);
        acc = _mm512_add_epi32(acc, prod_lo);

        // Process high 32 bytes
        let va_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(va, 1));
        let vb_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vb, 1));
        let prod_hi = _mm512_madd_epi16(va_hi, vb_hi);
        acc = _mm512_add_epi32(acc, prod_hi);
    }

    let mut result = hsum_epi32_512(acc);

    // Scalar tail
    for i in 0..remainder {
        let offset = chunks * I8_LANES + i;
        result += (*a.add(offset) as i32) * (*b.add(offset) as i32);
    }

    result
}
