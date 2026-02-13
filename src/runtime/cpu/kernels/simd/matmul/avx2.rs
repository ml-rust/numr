//! AVX2+FMA optimized matmul microkernels
//!
//! This module provides 256-bit SIMD microkernels for matrix multiplication.
//! These work on CPUs with AVX2 and FMA support (Intel Haswell+, AMD Excavator+).
//!
//! # Microkernel Dimensions
//!
//! - f32: 6x8 (6 rows × 8 columns = 48 elements per microkernel invocation)
//! - f64: 6x4 (6 rows × 4 columns = 24 elements per microkernel invocation)
//!
//! # Register Usage (f32 6x8)
//!
//! - ymm0-ymm5: C accumulators (6 rows × 8 columns)
//! - ymm6: A broadcast register
//! - ymm7: B load register
//!
//! This leaves ymm8-ymm15 available for potential loop unrolling or prefetching.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::macros::{
    define_microkernel_2x_f32, define_microkernel_2x_f64, define_microkernel_f32,
    define_microkernel_f64,
};

// Generate f32 6x8 microkernel using AVX2+FMA
define_microkernel_f32!(
    microkernel_6x8_f32,
    8,
    "avx2",
    "fma",
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    _mm256_set1_ps,
    _mm256_fmadd_ps,
    _mm256_setzero_ps,
    __m256
);

// Generate f64 6x4 microkernel using AVX2+FMA
define_microkernel_f64!(
    microkernel_6x4_f64,
    4,
    "avx2",
    "fma",
    _mm256_loadu_pd,
    _mm256_storeu_pd,
    _mm256_set1_pd,
    _mm256_fmadd_pd,
    _mm256_setzero_pd,
    __m256d
);

// Generate f32 6x16 double-width microkernel using AVX2+FMA (12 FMA chains)
define_microkernel_2x_f32!(
    microkernel_6x16_f32,
    8,
    "avx2",
    "fma",
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    _mm256_set1_ps,
    _mm256_fmadd_ps,
    _mm256_setzero_ps,
    __m256
);

// Generate f64 6x8 double-width microkernel using AVX2+FMA (12 FMA chains)
define_microkernel_2x_f64!(
    microkernel_6x8_f64,
    4,
    "avx2",
    "fma",
    _mm256_loadu_pd,
    _mm256_storeu_pd,
    _mm256_set1_pd,
    _mm256_fmadd_pd,
    _mm256_setzero_pd,
    __m256d
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_microkernel_6x8_f32_basic() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2+FMA");
            return;
        }

        // A: 6x2 matrix (packed for microkernel)
        let a: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // k=0
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // k=1
        ];

        // B: 2x8 matrix (packed row-major)
        let mut b: Vec<f32> = vec![0.0; 16];
        for i in 0..8 {
            b[i] = 1.0; // k=0, all ones
            b[8 + i] = (i + 1) as f32; // k=1, 1..8
        }

        // C: 6x8 matrix (initialize to zero)
        let mut c: Vec<f32> = vec![0.0; 6 * 8];

        unsafe {
            microkernel_6x8_f32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 8, true);
        }

        // Expected: C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j]
        //                   = (i+1) + 1*(j+1)
        //                   = i + j + 2
        for i in 0..6 {
            for j in 0..8 {
                let expected = (i + 1) as f32 + (j + 1) as f32;
                let actual = c[i * 8 + j];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "Mismatch at [{i}][{j}]: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn test_microkernel_6x4_f64_basic() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2+FMA");
            return;
        }

        let a: Vec<f64> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // k=0
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // k=1
        ];

        let mut b: Vec<f64> = vec![0.0; 8];
        for i in 0..4 {
            b[i] = 1.0;
            b[4 + i] = (i + 1) as f64;
        }

        let mut c: Vec<f64> = vec![0.0; 6 * 4];

        unsafe {
            microkernel_6x4_f64(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 4, true);
        }

        for i in 0..6 {
            for j in 0..4 {
                let expected = (i + 1) as f64 + (j + 1) as f64;
                let actual = c[i * 4 + j];
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "Mismatch at [{i}][{j}]: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn test_microkernel_accumulate() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test - CPU doesn't support AVX2+FMA");
            return;
        }

        let a: Vec<f32> = vec![1.0; 12]; // 6x2, all ones
        let b: Vec<f32> = vec![1.0; 16]; // 2x8, all ones

        // Initialize C to 100.0
        let mut c: Vec<f32> = vec![100.0; 6 * 8];

        unsafe {
            // Use accumulating version (first_k=false, beta=1)
            microkernel_6x8_f32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 8, false);
        }

        // Expected: C[i][j] = 100 + 2*1 = 102
        for i in 0..6 {
            for j in 0..8 {
                let expected = 102.0f32;
                let actual = c[i * 8 + j];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "Mismatch at [{i}][{j}]: expected {expected}, got {actual}"
                );
            }
        }
    }
}
