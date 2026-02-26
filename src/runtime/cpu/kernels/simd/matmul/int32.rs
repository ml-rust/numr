//! SIMD-optimized i32 matrix multiplication
//!
//! Uses AVX2 `_mm256_mullo_epi32` for 8-wide i32 multiply-accumulate.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::{SimdLevel, detect_simd};

/// SIMD-optimized i32 matrix multiplication: C = A @ B
///
/// # Safety
/// - All pointers must be valid for the specified dimensions
/// - `out` must not alias with `a` or `b`
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_i32(
    a: *const i32,
    b: *const i32,
    out: *mut i32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    let level = detect_simd();

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 | SimdLevel::Avx2Fma => {
            matmul_i32_avx2(a, b, out, m, n, k, lda, ldb, ldc);
            return;
        }
        _ => {}
    }

    // Scalar fallback
    #[cfg(target_arch = "aarch64")]
    let _ = level;

    matmul_i32_scalar(a, b, out, m, n, k, lda, ldb, ldc);
}

/// AVX2 i32 matmul: row × column with 8-wide multiply-accumulate
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_i32_avx2(
    a: *const i32,
    b: *const i32,
    out: *mut i32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    const LANES: usize = 8;

    for i in 0..m {
        let a_row = a.add(i * lda);

        // Process 8 output columns at a time
        let mut j = 0;
        while j + LANES <= n {
            let mut acc = _mm256_setzero_si256();

            for kk in 0..k {
                let a_val = _mm256_set1_epi32(*a_row.add(kk));
                let b_vals = _mm256_loadu_si256(b.add(kk * ldb + j) as *const __m256i);
                let prod = _mm256_mullo_epi32(a_val, b_vals);
                acc = _mm256_add_epi32(acc, prod);
            }

            _mm256_storeu_si256(out.add(i * ldc + j) as *mut __m256i, acc);
            j += LANES;
        }

        // Scalar tail for remaining columns
        while j < n {
            let mut sum = 0i32;
            for kk in 0..k {
                sum += (*a_row.add(kk)) * (*b.add(kk * ldb + j));
            }
            *out.add(i * ldc + j) = sum;
            j += 1;
        }
    }
}

/// Scalar i32 matmul fallback
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_i32_scalar(
    a: *const i32,
    b: *const i32,
    out: *mut i32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    // Zero output
    for i in 0..m {
        for j in 0..n {
            *out.add(i * ldc + j) = 0;
        }
    }

    // ikj order for cache locality
    for i in 0..m {
        for kk in 0..k {
            let a_val = *a.add(i * lda + kk);
            for j in 0..n {
                let out_ptr = out.add(i * ldc + j);
                *out_ptr += a_val * (*b.add(kk * ldb + j));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_i32_basic() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = [[19, 22], [43, 50]]
        let a = [1i32, 2, 3, 4];
        let b = [5i32, 6, 7, 8];
        let mut c = [0i32; 4];

        unsafe { matmul_i32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 2, 2, 2, 2, 2) };
        assert_eq!(c, [19, 22, 43, 50]);
    }

    #[test]
    fn test_matmul_i32_non_square() {
        // A(3x2) @ B(2x4) = C(3x4)
        let a = [1i32, 2, 3, 4, 5, 6];
        let b = [1i32, 2, 3, 4, 5, 6, 7, 8];
        let mut c = [0i32; 12];

        unsafe { matmul_i32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 3, 4, 2, 2, 4, 4) };
        assert_eq!(c, [11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68]);
    }

    #[test]
    fn test_matmul_i32_wide() {
        // Test with n > 8 to exercise SIMD path
        let (m, n, k) = (2, 16, 3);
        let a: Vec<i32> = (0..m * k).map(|i| (i + 1) as i32).collect();
        let b: Vec<i32> = (0..k * n).map(|i| (i + 1) as i32).collect();
        let mut c = vec![0i32; m * n];

        unsafe { matmul_i32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, k, n, n) };

        // Reference
        let mut expected = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                for kk in 0..k {
                    expected[i * n + j] += a[i * k + kk] * b[kk * n + j];
                }
            }
        }
        assert_eq!(c, expected);
    }
}
