//! Small-matrix SIMD matmul with register blocking
//!
//! For matrices below the tiling threshold, packing cost dominates.
//! These kernels use register-blocked SIMD FMA directly on unpacked row-major data.
//!
//! # Register Blocking Strategy
//!
//! Process MR_SMALL rows × 2 column chunks simultaneously:
//! - 4 rows × 2 chunks = 8 independent FMA accumulator chains
//! - FMA latency=4, throughput=0.5 → need 8 chains to saturate pipeline
//! - Each k iteration: 1 B load shared across 4 rows, 4 A broadcasts (1 per row)
//! - Outer product style: A broadcast × B vector → accumulate
//!
//! Kernel implementations are in `small_kernels.rs`, this file provides dispatch.

use super::small_kernels::*;
use crate::runtime::cpu::kernels::simd::SimdLevel;

#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn small_matmul_f32(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
) {
    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => small_matmul_f32_avx512(a, b, out, m, n, k, lda, ldb, ldc),
        SimdLevel::Avx2Fma => small_matmul_f32_avx2(a, b, out, m, n, k, lda, ldb, ldc),
        _ => super::scalar::matmul_scalar_f32(a, b, out, m, n, k, lda, ldb, ldc),
    }
    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            small_matmul_f32_neon(a, b, out, m, n, k, lda, ldb, ldc)
        }
        _ => super::scalar::matmul_scalar_f32(a, b, out, m, n, k, lda, ldb, ldc),
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = level;
        super::scalar::matmul_scalar_f32(a, b, out, m, n, k, lda, ldb, ldc);
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn small_matmul_f64(
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
) {
    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => small_matmul_f64_avx512(a, b, out, m, n, k, lda, ldb, ldc),
        SimdLevel::Avx2Fma => small_matmul_f64_avx2(a, b, out, m, n, k, lda, ldb, ldc),
        _ => super::scalar::matmul_scalar_f64(a, b, out, m, n, k, lda, ldb, ldc),
    }
    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            small_matmul_f64_neon(a, b, out, m, n, k, lda, ldb, ldc)
        }
        _ => super::scalar::matmul_scalar_f64(a, b, out, m, n, k, lda, ldb, ldc),
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = level;
        super::scalar::matmul_scalar_f64(a, b, out, m, n, k, lda, ldb, ldc);
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn small_matmul_bias_f32(
    a: *const f32,
    b: *const f32,
    bias: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
) {
    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => small_matmul_bias_f32_avx512(a, b, bias, out, m, n, k, lda, ldb, ldc),
        SimdLevel::Avx2Fma => small_matmul_bias_f32_avx2(a, b, bias, out, m, n, k, lda, ldb, ldc),
        _ => super::scalar::matmul_bias_scalar_f32(a, b, bias, out, m, n, k, lda, ldb, ldc),
    }
    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            small_matmul_bias_f32_neon(a, b, bias, out, m, n, k, lda, ldb, ldc)
        }
        _ => super::scalar::matmul_bias_scalar_f32(a, b, bias, out, m, n, k, lda, ldb, ldc),
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = level;
        super::scalar::matmul_bias_scalar_f32(a, b, bias, out, m, n, k, lda, ldb, ldc);
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn small_matmul_bias_f64(
    a: *const f64,
    b: *const f64,
    bias: *const f64,
    out: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
) {
    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => small_matmul_bias_f64_avx512(a, b, bias, out, m, n, k, lda, ldb, ldc),
        SimdLevel::Avx2Fma => small_matmul_bias_f64_avx2(a, b, bias, out, m, n, k, lda, ldb, ldc),
        _ => super::scalar::matmul_bias_scalar_f64(a, b, bias, out, m, n, k, lda, ldb, ldc),
    }
    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            small_matmul_bias_f64_neon(a, b, bias, out, m, n, k, lda, ldb, ldc)
        }
        _ => super::scalar::matmul_bias_scalar_f64(a, b, bias, out, m, n, k, lda, ldb, ldc),
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = level;
        super::scalar::matmul_bias_scalar_f64(a, b, bias, out, m, n, k, lda, ldb, ldc);
    }
}
