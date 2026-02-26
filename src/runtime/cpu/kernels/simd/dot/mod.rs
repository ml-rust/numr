//! SIMD-accelerated integer dot product operations
//!
//! Provides high-throughput i8 x i8 → i32 dot products for quantized inference.
//!
//! # Architecture Support
//!
//! | Architecture | Instruction Set  | Elements/cycle | Key Intrinsic          |
//! |--------------|------------------|----------------|------------------------|
//! | x86-64       | AVX-512BW        | 64             | maddubs + madd         |
//! | x86-64       | AVX2             | 32             | maddubs + madd         |
//! | ARM64        | NEON             | 16             | vmull_s8 + vpadalq_s16 |

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use super::{SimdLevel, detect_simd};

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
        SimdLevel::Avx512 => {
            if is_x86_feature_detected!("avx512bw") {
                return x86_64::avx512::i8xi8_dot_i32(a, b, len);
            }
            return x86_64::avx2::i8xi8_dot_i32(a, b, len);
        }
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
pub unsafe fn i8xi8_dot_f32(a: *const i8, b: *const i8, scale: f32, len: usize) -> f32 {
    (i8xi8_dot_i32(a, b, len) as f32) * scale
}

/// Scalar fallback for i8 dot product
#[inline]
unsafe fn i8xi8_dot_scalar(a: *const i8, b: *const i8, len: usize) -> i32 {
    let mut acc = 0i32;
    for i in 0..len {
        acc += (*a.add(i) as i32) * (*b.add(i) as i32);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

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
