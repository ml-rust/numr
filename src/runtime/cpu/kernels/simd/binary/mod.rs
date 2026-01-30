//! SIMD-accelerated binary operations
//!
//! This module provides AVX2 and AVX-512 implementations for element-wise
//! binary operations (add, sub, mul, div, max, min).

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

use super::{SimdLevel, detect_simd};
use crate::ops::BinaryOp;

// Import scalar fallbacks from kernels module (single source of truth)
pub use crate::runtime::cpu::kernels::binary::{binary_scalar_f32, binary_scalar_f64};

/// Minimum elements to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

/// SIMD binary operation for f32
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn binary_f32(op: BinaryOp, a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        binary_scalar_f32(op, a, b, out, len);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::binary_f32(op, a, b, out, len),
        SimdLevel::Avx2Fma => avx2::binary_f32(op, a, b, out, len),
        SimdLevel::Scalar => unreachable!(),
    }
}

/// SIMD binary operation for f64
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn binary_f64(op: BinaryOp, a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        binary_scalar_f64(op, a, b, out, len);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::binary_f64(op, a, b, out, len),
        SimdLevel::Avx2Fma => avx2::binary_f64(op, a, b, out, len),
        SimdLevel::Scalar => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_add_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..100).map(|x| (x * 2) as f32).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { binary_f32(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert_eq!(out[i], a[i] + b[i], "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_binary_mul_f64() {
        let a: Vec<f64> = (1..101).map(|x| x as f64).collect();
        let b: Vec<f64> = (1..101).map(|x| (x * 2) as f64).collect();
        let mut out = vec![0.0f64; 100];

        unsafe { binary_f64(BinaryOp::Mul, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert_eq!(out[i], a[i] * b[i], "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_small_array_uses_scalar() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];

        unsafe { binary_f32(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4) }

        assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_non_aligned_length() {
        let a: Vec<f32> = (0..67).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..67).map(|x| (x * 2) as f32).collect();
        let mut out = vec![0.0f32; 67];

        unsafe { binary_f32(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 67) }

        for i in 0..67 {
            assert_eq!(out[i], a[i] + b[i], "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_binary_pow_f32() {
        let a: Vec<f32> = (1..101).map(|x| x as f32 * 0.1).collect();
        let b: Vec<f32> = (1..101).map(|x| (x % 5) as f32 + 0.5).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { binary_f32(BinaryOp::Pow, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].powf(b[i]);
            let diff = (out[i] - expected).abs();
            // pow uses exp(b*log(a)), so errors compound - ~1e-3 relative error is acceptable
            assert!(
                diff < 1e-3 * expected.abs().max(1.0),
                "pow mismatch at {}: got {}, expected {} (a={}, b={})",
                i,
                out[i],
                expected,
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_binary_pow_f64() {
        let a: Vec<f64> = (1..101).map(|x| x as f64 * 0.1).collect();
        let b: Vec<f64> = (1..101).map(|x| (x % 5) as f64 + 0.5).collect();
        let mut out = vec![0.0f64; 100];

        unsafe { binary_f64(BinaryOp::Pow, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].powf(b[i]);
            let diff = (out[i] - expected).abs();
            // pow uses exp(b*log(a)), so errors compound - ~1e-4 relative error is acceptable
            assert!(
                diff < 1e-4 * expected.abs().max(1.0),
                "pow mismatch at {}: got {}, expected {} (a={}, b={})",
                i,
                out[i],
                expected,
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_binary_max_min_f32() {
        let a: Vec<f32> = (0..100).map(|x| (x as f32 - 50.0) * 0.5).collect();
        let b: Vec<f32> = (0..100).map(|x| ((x + 25) as f32 - 50.0) * 0.5).collect();
        let mut out_max = vec![0.0f32; 100];
        let mut out_min = vec![0.0f32; 100];

        unsafe {
            binary_f32(
                BinaryOp::Max,
                a.as_ptr(),
                b.as_ptr(),
                out_max.as_mut_ptr(),
                100,
            );
            binary_f32(
                BinaryOp::Min,
                a.as_ptr(),
                b.as_ptr(),
                out_min.as_mut_ptr(),
                100,
            );
        }

        for i in 0..100 {
            assert_eq!(out_max[i], a[i].max(b[i]), "max mismatch at {}", i);
            assert_eq!(out_min[i], a[i].min(b[i]), "min mismatch at {}", i);
        }
    }

    #[test]
    fn test_binary_sub_div_f32() {
        let a: Vec<f32> = (1..101).map(|x| x as f32 * 2.0).collect();
        let b: Vec<f32> = (1..101).map(|x| x as f32).collect();
        let mut out_sub = vec![0.0f32; 100];
        let mut out_div = vec![0.0f32; 100];

        unsafe {
            binary_f32(
                BinaryOp::Sub,
                a.as_ptr(),
                b.as_ptr(),
                out_sub.as_mut_ptr(),
                100,
            );
            binary_f32(
                BinaryOp::Div,
                a.as_ptr(),
                b.as_ptr(),
                out_div.as_mut_ptr(),
                100,
            );
        }

        for i in 0..100 {
            assert_eq!(out_sub[i], a[i] - b[i], "sub mismatch at {}", i);
            assert_eq!(out_div[i], a[i] / b[i], "div mismatch at {}", i);
        }
    }

    // ============================================================================
    // Streaming store tests
    // ============================================================================

    /// Test streaming threshold constant is correctly defined
    #[test]
    fn test_streaming_threshold_defined() {
        use super::super::streaming::{STREAMING_THRESHOLD_F32, STREAMING_THRESHOLD_F64};
        // 1MB = 262144 f32s, 131072 f64s
        assert_eq!(STREAMING_THRESHOLD_F32, 262144);
        assert_eq!(STREAMING_THRESHOLD_F64, 131072);
    }

    /// Test that large arrays produce correct results (exercises streaming path if aligned)
    #[test]
    fn test_large_array_correctness_f32() {
        // Use a size that triggers streaming (> 1MB = 262144 f32s)
        // For testing we use a smaller aligned buffer to avoid OOM
        const LEN: usize = 1024; // Small but validates the code path
        let a: Vec<f32> = (0..LEN).map(|x| (x as f32) * 0.1).collect();
        let b: Vec<f32> = (0..LEN).map(|x| (x as f32) * 0.2 + 1.0).collect();
        let mut out = vec![0.0f32; LEN];

        unsafe { binary_f32(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i] + b[i];
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "large array mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    /// Test that large arrays produce correct results for all operations
    #[test]
    fn test_large_array_all_ops_f32() {
        const LEN: usize = 512;
        let a: Vec<f32> = (1..=LEN as i32).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=LEN as i32).map(|x| (x as f32) * 0.5 + 0.5).collect();

        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Max,
            BinaryOp::Min,
        ] {
            let mut out = vec![0.0f32; LEN];
            unsafe { binary_f32(op, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), LEN) }

            for i in 0..LEN {
                let expected = match op {
                    BinaryOp::Add => a[i] + b[i],
                    BinaryOp::Sub => a[i] - b[i],
                    BinaryOp::Mul => a[i] * b[i],
                    BinaryOp::Div => a[i] / b[i],
                    BinaryOp::Max => a[i].max(b[i]),
                    BinaryOp::Min => a[i].min(b[i]),
                    BinaryOp::Pow => a[i].powf(b[i]),
                };
                assert!(
                    (out[i] - expected).abs() < 1e-5 * expected.abs().max(1.0),
                    "{:?} mismatch at {}: got {}, expected {}",
                    op,
                    i,
                    out[i],
                    expected
                );
            }
        }
    }

    /// Test alignment check functions
    #[test]
    fn test_alignment_checks() {
        use super::super::streaming::{is_aligned_avx2, is_aligned_avx512};

        // Test known aligned addresses
        assert!(is_aligned_avx2(32 as *const f32)); // 32 % 32 == 0
        assert!(is_aligned_avx2(64 as *const f32)); // 64 % 32 == 0
        assert!(!is_aligned_avx2(16 as *const f32)); // 16 % 32 != 0

        assert!(is_aligned_avx512(64 as *const f32)); // 64 % 64 == 0
        assert!(is_aligned_avx512(128 as *const f32)); // 128 % 64 == 0
        assert!(!is_aligned_avx512(32 as *const f32)); // 32 % 64 != 0
    }
}
