//! SIMD dispatch entry points for i8xi8 dot products.

#[cfg(target_arch = "aarch64")]
use super::aarch64;
#[cfg(target_arch = "x86_64")]
use super::x86_64;

use super::super::{SimdLevel, detect_simd};
use super::scalar::i8xi8_dot_scalar;

/// Minimum elements to justify SIMD overhead for dot products
const DOT_SIMD_THRESHOLD: usize = 32;

/// Dot product of signed i8 vectors, accumulated in i32.
///
/// Automatically dispatches to the best SIMD implementation available:
/// - x86-64/AVX-512BW: 64 elements per iteration via `_mm512_maddubs_epi16` + `_mm512_madd_epi16`
/// - x86-64/AVX2: 32 elements per iteration via `_mm256_maddubs_epi16` + `_mm256_madd_epi16`
/// - ARM64/NEON: 16 elements per iteration via `vmull_s8` + `vpadalq_s16`
/// - Scalar fallback for small arrays (<32 elements) or unsupported platforms
///
/// Computes sum(a[i] * b[i]) for i in 0..len.
///
/// # Safety
/// - `a` and `b` must be valid pointers to `len` elements
#[inline]
pub unsafe fn i8xi8_dot_i32(a: *const i8, b: *const i8, len: usize) -> i32 {
    let level = detect_simd();

    if len < DOT_SIMD_THRESHOLD || level == SimdLevel::Scalar {
        return i8xi8_dot_scalar(a, b, len);
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => return x86_64::avx512::i8xi8_dot_i32(a, b, len),
        SimdLevel::Avx2Fma => return x86_64::avx2::i8xi8_dot_i32(a, b, len),
        _ => return i8xi8_dot_scalar(a, b, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => return aarch64::neon::i8xi8_dot_i32(a, b, len),
        _ => return i8xi8_dot_scalar(a, b, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    i8xi8_dot_scalar(a, b, len)
}

/// Scaled dot product of signed i8 vectors, returning f32.
///
/// Computes scale * sum(a[i] * b[i]) for i in 0..len.
///
/// # Safety
/// - `a` and `b` must be valid pointers to `len` elements
#[inline]
#[allow(dead_code)] // Public API for downstream crates (e.g., boostr quantized ops)
pub unsafe fn i8xi8_dot_f32(a: *const i8, b: *const i8, scale: f32, len: usize) -> f32 {
    (i8xi8_dot_i32(a, b, len) as f32) * scale
}

#[cfg(test)]
mod tests {
    use super::super::scalar::{DOT_SPILL_ITERS, i8xi8_dot_scalar, saturate_i64_to_i32};
    use super::*;

    /// Longest run where an i32 accumulator is still exact: `i32::MAX / 128^2`.
    const I32_ACC_LIMIT: usize = (i32::MAX as usize) / (128 * 128);

    #[test]
    fn saturation_clamps_instead_of_wrapping() {
        assert_eq!(saturate_i64_to_i32(0), 0);
        assert_eq!(saturate_i64_to_i32(i32::MAX as i64), i32::MAX);
        assert_eq!(saturate_i64_to_i32(i32::MAX as i64 + 1), i32::MAX);
        assert_eq!(saturate_i64_to_i32(i32::MIN as i64 - 1), i32::MIN);
        assert_eq!(saturate_i64_to_i32(i64::MAX), i32::MAX);
        assert_eq!(saturate_i64_to_i32(i64::MIN), i32::MIN);
    }

    // Why there is no separate test for the periodic lane spill:
    //
    // A wrapped i32 lane is only OBSERVABLE when the final total leaves i32 range.
    // Two's-complement addition is a ring, so a lane that wraps by `+X` and later
    // has `X` subtracted lands back on the correct value — a test built from
    // cancelling halves passes whether or not the spill exists. The two saturation
    // tests below are the ones that catch a missing spill, because there the wrong
    // lane value reaches the clamp and comes out as the wrong bound.
    //
    // Verified by deleting the AVX2 spill: those two fail, and a cancelling-halves
    // test does not.

    #[test]
    fn a_long_dot_product_saturates_rather_than_changing_sign() {
        // 127 * 127 * len overflows i32 once len passes ~133k. With an i32
        // accumulator this returned a NEGATIVE number for a product of two
        // all-positive vectors.
        // Long enough to cross several spill periods, not just the i32 accumulator
        // limit — a length under one period leaves the spill untested.
        let len = (8 * I32_ACC_LIMIT).max(3 * DOT_SPILL_ITERS * 32);
        let a = vec![127i8; len];
        let b = vec![127i8; len];

        let got = unsafe { i8xi8_dot_i32(a.as_ptr(), b.as_ptr(), len) };
        let exact: i64 = 127 * 127 * len as i64;
        assert!(exact > i32::MAX as i64, "test input must actually overflow");
        assert_eq!(got, i32::MAX, "expected clamp, got {got}");
        assert!(
            got > 0,
            "an all-positive dot product must not come back negative"
        );
    }

    #[test]
    fn the_scalar_path_saturates_the_same_way() {
        // The dispatcher sends long inputs to SIMD, so the scalar fallback needs
        // its own coverage — it has the same i32 accumulator bug.
        let len = 4 * I32_ACC_LIMIT;
        let a = vec![127i8; len];
        let b = vec![127i8; len];
        let got = unsafe { i8xi8_dot_scalar(a.as_ptr(), b.as_ptr(), len) };
        assert_eq!(got, i32::MAX, "expected clamp, got {got}");
    }

    #[test]
    fn negative_overflow_clamps_to_min() {
        let len = (8 * I32_ACC_LIMIT).max(3 * DOT_SPILL_ITERS * 32);
        let a = vec![127i8; len];
        let b = vec![-127i8; len];
        let got = unsafe { i8xi8_dot_i32(a.as_ptr(), b.as_ptr(), len) };
        assert_eq!(got, i32::MIN, "expected clamp, got {got}");
        assert!(got < 0);
    }

    #[test]
    fn a_long_dot_product_that_fits_is_still_exact() {
        // Saturation must not fire early. This length crosses several spill
        // boundaries but its total stays inside i32, so the answer is exact.
        let len = 3 * DOT_SPILL_ITERS * 32 + 7;
        let a: Vec<i8> = (0..len).map(|i| ((i % 5) as i8) - 2).collect();
        let b: Vec<i8> = (0..len).map(|i| ((i % 7) as i8) - 3).collect();

        let expected: i64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as i64 * y as i64)
            .sum();
        assert!(
            expected.abs() < i32::MAX as i64,
            "this case must NOT saturate"
        );

        let got = unsafe { i8xi8_dot_i32(a.as_ptr(), b.as_ptr(), len) };
        assert_eq!(got as i64, expected);
    }

    #[test]
    fn simd_and_scalar_agree_across_lengths_that_straddle_the_spill() {
        // Every backend spills its lane accumulator on a fixed period. A spill that
        // drops or double-counts a block shows up as a mismatch here, at lengths
        // just below, on, and just above the boundary.
        for len in [
            1,
            31,
            32,
            33,
            DOT_SPILL_ITERS * 32 - 1,
            DOT_SPILL_ITERS * 32,
            DOT_SPILL_ITERS * 32 + 1,
            DOT_SPILL_ITERS * 64 + 17,
        ] {
            let a: Vec<i8> = (0..len).map(|i| ((i % 11) as i8) - 5).collect();
            let b: Vec<i8> = (0..len).map(|i| ((i % 13) as i8) - 6).collect();
            let simd = unsafe { i8xi8_dot_i32(a.as_ptr(), b.as_ptr(), len) };
            let scalar = unsafe { i8xi8_dot_scalar(a.as_ptr(), b.as_ptr(), len) };
            assert_eq!(simd, scalar, "len {len}");
        }
    }

    #[test]
    fn test_i8xi8_dot_basic() {
        let a: Vec<i8> = (0..100).map(|x| (x % 127) as i8).collect();
        let b: Vec<i8> = (0..100).map(|x| ((x * 3) % 127) as i8).collect();

        let result = unsafe { i8xi8_dot_i32(a.as_ptr(), b.as_ptr(), a.len()) };

        // Compute expected
        let expected: i32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_i8xi8_dot_negative() {
        let a: Vec<i8> = (0..64).map(|x| (x as i8) - 32).collect();
        let b: Vec<i8> = (0..64).map(|x| (x as i8) - 16).collect();

        let result = unsafe { i8xi8_dot_i32(a.as_ptr(), b.as_ptr(), a.len()) };
        let expected: i32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_i8xi8_dot_tail() {
        // Non-aligned length to exercise scalar tail
        let a: Vec<i8> = (0..67).map(|x| (x % 50) as i8).collect();
        let b: Vec<i8> = (0..67).map(|x| ((x * 2) % 50) as i8).collect();

        let result = unsafe { i8xi8_dot_i32(a.as_ptr(), b.as_ptr(), a.len()) };
        let expected: i32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_i8xi8_dot_small() {
        let a: Vec<i8> = vec![1, 2, 3, 4];
        let b: Vec<i8> = vec![5, 6, 7, 8];

        let result = unsafe { i8xi8_dot_i32(a.as_ptr(), b.as_ptr(), a.len()) };
        assert_eq!(result, 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8);
    }

    #[test]
    fn test_i8xi8_dot_f32_scaled() {
        let a: Vec<i8> = vec![10, 20, 30, 40];
        let b: Vec<i8> = vec![1, 2, 3, 4];
        let scale = 0.5f32;

        let result = unsafe { i8xi8_dot_f32(a.as_ptr(), b.as_ptr(), scale, a.len()) };
        let expected = (10 + 40 + 90 + 160) as f32 * scale;
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_i8xi8_dot_extremes() {
        // Test with extreme i8 values
        let a: Vec<i8> = vec![
            -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127,
            -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127,
        ];
        let b: Vec<i8> = vec![
            127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128,
            127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128,
        ];

        let result = unsafe { i8xi8_dot_i32(a.as_ptr(), b.as_ptr(), a.len()) };
        let expected: i32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_i8xi8_dot_large() {
        let a: Vec<i8> = (0..1024)
            .map(|x| ((x * 7 + 13) % 256 - 128) as i8)
            .collect();
        let b: Vec<i8> = (0..1024)
            .map(|x| ((x * 11 + 5) % 256 - 128) as i8)
            .collect();

        let result = unsafe { i8xi8_dot_i32(a.as_ptr(), b.as_ptr(), a.len()) };
        let expected: i32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum();
        assert_eq!(result, expected);
    }
}
