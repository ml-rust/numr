//! AVX-512 optimized matmul microkernels
//!
//! This module provides 512-bit SIMD microkernels for matrix multiplication.
//! These are the innermost loops of the tiled matmul algorithm.
//!
//! # Microkernel Dimensions
//!
//! - f32: 6x16 (6 rows × 16 columns = 96 elements per microkernel invocation)
//! - f64: 6x8 (6 rows × 8 columns = 48 elements per microkernel invocation)
//!
//! # Register Usage (f32 6x16)
//!
//! - zmm0-zmm5: C accumulators (6 rows × 16 columns)
//! - zmm6: A broadcast register
//! - zmm7: B load register

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::macros::{define_microkernel_f32, define_microkernel_f64};

// Generate f32 6x16 microkernel using AVX-512
define_microkernel_f32!(
    microkernel_6x16_f32,
    16,
    "avx512f",
    "fma",
    _mm512_loadu_ps,
    _mm512_storeu_ps,
    _mm512_set1_ps,
    _mm512_fmadd_ps,
    __m512
);

// Generate f64 6x8 microkernel using AVX-512
define_microkernel_f64!(
    microkernel_6x8_f64,
    8,
    "avx512f",
    "fma",
    _mm512_loadu_pd,
    _mm512_storeu_pd,
    _mm512_set1_pd,
    _mm512_fmadd_pd,
    __m512d
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_microkernel_6x16_f32_basic() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX-512 test - CPU doesn't support AVX-512F+FMA");
            return;
        }

        // A: 6x2 packed, format: [a00,a10,a20,a30,a40,a50, a01,a11,a21,a31,a41,a51]
        let a: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // k=0
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // k=1
        ];

        // B: 2x16 packed row-major
        let mut b: Vec<f32> = vec![0.0; 32];
        for i in 0..16 {
            b[i] = 1.0;
            b[16 + i] = (i + 1) as f32;
        }

        // Initialize C to zero (simulates beta=0)
        let mut c: Vec<f32> = vec![0.0; 6 * 16];

        unsafe {
            microkernel_6x16_f32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 16);
        }

        // C[i][j] = A[i][0]*1 + A[i][1]*(j+1) = (i+1) + (j+1)
        for i in 0..6 {
            for j in 0..16 {
                let expected = (i + 1) as f32 + (j + 1) as f32;
                let actual = c[i * 16 + j];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "Mismatch at [{i}][{j}]: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn test_microkernel_6x8_f64_basic() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX-512 test - CPU doesn't support AVX-512F+FMA");
            return;
        }

        let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut b: Vec<f64> = vec![0.0; 16];
        for i in 0..8 {
            b[i] = 1.0;
            b[8 + i] = (i + 1) as f64;
        }

        let mut c: Vec<f64> = vec![0.0; 6 * 8];

        unsafe {
            microkernel_6x8_f64(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 8);
        }

        for i in 0..6 {
            for j in 0..8 {
                let expected = (i + 1) as f64 + (j + 1) as f64;
                let actual = c[i * 8 + j];
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "Mismatch at [{i}][{j}]: expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn test_microkernel_accumulate() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX-512 test - CPU doesn't support AVX-512F+FMA");
            return;
        }

        let a: Vec<f32> = vec![1.0; 12]; // 6x2, all ones
        let b: Vec<f32> = vec![1.0; 32]; // 2x16, all ones

        // Initialize C to 100.0 to test accumulation
        let mut c: Vec<f32> = vec![100.0; 6 * 16];

        unsafe {
            microkernel_6x16_f32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 16);
        }

        // Expected: C[i][j] = 100 + 2*1 = 102
        for i in 0..6 {
            for j in 0..16 {
                let expected = 102.0f32;
                let actual = c[i * 16 + j];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "Mismatch at [{i}][{j}]: expected {expected}, got {actual}"
                );
            }
        }
    }
}
