//! Architecture-specific register-blocked SIMD kernels for small matmul
//!
//! Contains macro definitions and instantiations for x86_64 (AVX2, AVX-512)
//! and aarch64 (NEON) register-blocked matmul kernels.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Number of rows to process simultaneously in the register-blocked kernel
pub(super) const MR_SMALL: usize = 4;

// ---------------------------------------------------------------------------
// x86_64 register-blocked matmul
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
macro_rules! define_small_matmul_regblocked_x86 {
    ($name:ident, $ty:ty, $W:expr, $feat1:literal, $feat2:literal,
     $loadu:ident, $storeu:ident, $set1:ident, $fmadd:ident, $setzero:ident, $vec:ty) => {
        #[target_feature(enable = $feat1, enable = $feat2)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            a: *const $ty,
            b: *const $ty,
            out: *mut $ty,
            m: usize,
            n: usize,
            k: usize,
            lda: usize,
            ldb: usize,
            ldc: usize,
        ) {
            let mr = MR_SMALL;
            let mut i = 0;

            // Main loop: process MR_SMALL rows at a time
            while i + mr <= m {
                let mut j = 0;

                // Process 2 column chunks simultaneously (2*W columns)
                while j + 2 * $W <= n {
                    // 8 accumulators: 4 rows × 2 column chunks
                    let mut c00: $vec = $setzero();
                    let mut c01: $vec = $setzero();
                    let mut c10: $vec = $setzero();
                    let mut c11: $vec = $setzero();
                    let mut c20: $vec = $setzero();
                    let mut c21: $vec = $setzero();
                    let mut c30: $vec = $setzero();
                    let mut c31: $vec = $setzero();

                    for kk in 0..k {
                        // Load 2 B vectors (shared across all 4 rows)
                        let b0 = $loadu(b.add(kk * ldb + j));
                        let b1 = $loadu(b.add(kk * ldb + j + $W));

                        // Row 0
                        let a0 = $set1(*a.add((i + 0) * lda + kk));
                        c00 = $fmadd(a0, b0, c00);
                        c01 = $fmadd(a0, b1, c01);

                        // Row 1
                        let a1 = $set1(*a.add((i + 1) * lda + kk));
                        c10 = $fmadd(a1, b0, c10);
                        c11 = $fmadd(a1, b1, c11);

                        // Row 2
                        let a2 = $set1(*a.add((i + 2) * lda + kk));
                        c20 = $fmadd(a2, b0, c20);
                        c21 = $fmadd(a2, b1, c21);

                        // Row 3
                        let a3 = $set1(*a.add((i + 3) * lda + kk));
                        c30 = $fmadd(a3, b0, c30);
                        c31 = $fmadd(a3, b1, c31);
                    }

                    // Store 8 results
                    $storeu(out.add((i + 0) * ldc + j), c00);
                    $storeu(out.add((i + 0) * ldc + j + $W), c01);
                    $storeu(out.add((i + 1) * ldc + j), c10);
                    $storeu(out.add((i + 1) * ldc + j + $W), c11);
                    $storeu(out.add((i + 2) * ldc + j), c20);
                    $storeu(out.add((i + 2) * ldc + j + $W), c21);
                    $storeu(out.add((i + 3) * ldc + j), c30);
                    $storeu(out.add((i + 3) * ldc + j + $W), c31);
                    j += 2 * $W;
                }

                // Remaining column chunks: 1 chunk at a time, still 4 rows
                while j + $W <= n {
                    let mut c0: $vec = $setzero();
                    let mut c1: $vec = $setzero();
                    let mut c2: $vec = $setzero();
                    let mut c3: $vec = $setzero();

                    for kk in 0..k {
                        let bv = $loadu(b.add(kk * ldb + j));
                        c0 = $fmadd($set1(*a.add((i + 0) * lda + kk)), bv, c0);
                        c1 = $fmadd($set1(*a.add((i + 1) * lda + kk)), bv, c1);
                        c2 = $fmadd($set1(*a.add((i + 2) * lda + kk)), bv, c2);
                        c3 = $fmadd($set1(*a.add((i + 3) * lda + kk)), bv, c3);
                    }

                    $storeu(out.add((i + 0) * ldc + j), c0);
                    $storeu(out.add((i + 1) * ldc + j), c1);
                    $storeu(out.add((i + 2) * ldc + j), c2);
                    $storeu(out.add((i + 3) * ldc + j), c3);
                    j += $W;
                }

                // Scalar tail columns
                while j < n {
                    let mut s0: $ty = 0.0;
                    let mut s1: $ty = 0.0;
                    let mut s2: $ty = 0.0;
                    let mut s3: $ty = 0.0;
                    for kk in 0..k {
                        let bv = *b.add(kk * ldb + j);
                        s0 += *a.add((i + 0) * lda + kk) * bv;
                        s1 += *a.add((i + 1) * lda + kk) * bv;
                        s2 += *a.add((i + 2) * lda + kk) * bv;
                        s3 += *a.add((i + 3) * lda + kk) * bv;
                    }
                    *out.add((i + 0) * ldc + j) = s0;
                    *out.add((i + 1) * ldc + j) = s1;
                    *out.add((i + 2) * ldc + j) = s2;
                    *out.add((i + 3) * ldc + j) = s3;
                    j += 1;
                }

                i += mr;
            }

            // Remaining rows: 1 row at a time
            while i < m {
                let mut j = 0;
                while j + $W <= n {
                    let mut acc: $vec = $setzero();
                    for kk in 0..k {
                        acc = $fmadd(
                            $set1(*a.add(i * lda + kk)),
                            $loadu(b.add(kk * ldb + j)),
                            acc,
                        );
                    }
                    $storeu(out.add(i * ldc + j), acc);
                    j += $W;
                }
                while j < n {
                    let mut sum: $ty = 0.0;
                    for kk in 0..k {
                        sum += *a.add(i * lda + kk) * *b.add(kk * ldb + j);
                    }
                    *out.add(i * ldc + j) = sum;
                    j += 1;
                }
                i += 1;
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
macro_rules! define_small_matmul_bias_regblocked_x86 {
    ($name:ident, $ty:ty, $W:expr, $feat1:literal, $feat2:literal,
     $loadu:ident, $storeu:ident, $set1:ident, $fmadd:ident, $setzero:ident, $vec:ty) => {
        #[target_feature(enable = $feat1, enable = $feat2)]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            a: *const $ty,
            b: *const $ty,
            bias: *const $ty,
            out: *mut $ty,
            m: usize,
            n: usize,
            k: usize,
            lda: usize,
            ldb: usize,
            ldc: usize,
        ) {
            let mr = MR_SMALL;
            let mut i = 0;

            while i + mr <= m {
                let mut j = 0;

                while j + 2 * $W <= n {
                    let bias0 = $loadu(bias.add(j));
                    let bias1 = $loadu(bias.add(j + $W));
                    let mut c00 = bias0;
                    let mut c01 = bias1;
                    let mut c10 = bias0;
                    let mut c11 = bias1;
                    let mut c20 = bias0;
                    let mut c21 = bias1;
                    let mut c30 = bias0;
                    let mut c31 = bias1;

                    for kk in 0..k {
                        let b0 = $loadu(b.add(kk * ldb + j));
                        let b1 = $loadu(b.add(kk * ldb + j + $W));

                        let a0 = $set1(*a.add((i + 0) * lda + kk));
                        c00 = $fmadd(a0, b0, c00);
                        c01 = $fmadd(a0, b1, c01);

                        let a1 = $set1(*a.add((i + 1) * lda + kk));
                        c10 = $fmadd(a1, b0, c10);
                        c11 = $fmadd(a1, b1, c11);

                        let a2 = $set1(*a.add((i + 2) * lda + kk));
                        c20 = $fmadd(a2, b0, c20);
                        c21 = $fmadd(a2, b1, c21);

                        let a3 = $set1(*a.add((i + 3) * lda + kk));
                        c30 = $fmadd(a3, b0, c30);
                        c31 = $fmadd(a3, b1, c31);
                    }

                    $storeu(out.add((i + 0) * ldc + j), c00);
                    $storeu(out.add((i + 0) * ldc + j + $W), c01);
                    $storeu(out.add((i + 1) * ldc + j), c10);
                    $storeu(out.add((i + 1) * ldc + j + $W), c11);
                    $storeu(out.add((i + 2) * ldc + j), c20);
                    $storeu(out.add((i + 2) * ldc + j + $W), c21);
                    $storeu(out.add((i + 3) * ldc + j), c30);
                    $storeu(out.add((i + 3) * ldc + j + $W), c31);
                    j += 2 * $W;
                }

                while j + $W <= n {
                    let biasv = $loadu(bias.add(j));
                    let mut c0 = biasv;
                    let mut c1 = biasv;
                    let mut c2 = biasv;
                    let mut c3 = biasv;

                    for kk in 0..k {
                        let bv = $loadu(b.add(kk * ldb + j));
                        c0 = $fmadd($set1(*a.add((i + 0) * lda + kk)), bv, c0);
                        c1 = $fmadd($set1(*a.add((i + 1) * lda + kk)), bv, c1);
                        c2 = $fmadd($set1(*a.add((i + 2) * lda + kk)), bv, c2);
                        c3 = $fmadd($set1(*a.add((i + 3) * lda + kk)), bv, c3);
                    }

                    $storeu(out.add((i + 0) * ldc + j), c0);
                    $storeu(out.add((i + 1) * ldc + j), c1);
                    $storeu(out.add((i + 2) * ldc + j), c2);
                    $storeu(out.add((i + 3) * ldc + j), c3);
                    j += $W;
                }

                while j < n {
                    let bval = *bias.add(j);
                    let mut s0 = bval;
                    let mut s1 = bval;
                    let mut s2 = bval;
                    let mut s3 = bval;
                    for kk in 0..k {
                        let bv = *b.add(kk * ldb + j);
                        s0 += *a.add((i + 0) * lda + kk) * bv;
                        s1 += *a.add((i + 1) * lda + kk) * bv;
                        s2 += *a.add((i + 2) * lda + kk) * bv;
                        s3 += *a.add((i + 3) * lda + kk) * bv;
                    }
                    *out.add((i + 0) * ldc + j) = s0;
                    *out.add((i + 1) * ldc + j) = s1;
                    *out.add((i + 2) * ldc + j) = s2;
                    *out.add((i + 3) * ldc + j) = s3;
                    j += 1;
                }

                i += mr;
            }

            // Remaining rows
            while i < m {
                let mut j = 0;
                while j + $W <= n {
                    let mut acc = $loadu(bias.add(j));
                    for kk in 0..k {
                        acc = $fmadd(
                            $set1(*a.add(i * lda + kk)),
                            $loadu(b.add(kk * ldb + j)),
                            acc,
                        );
                    }
                    $storeu(out.add(i * ldc + j), acc);
                    j += $W;
                }
                while j < n {
                    let mut sum = *bias.add(j);
                    for kk in 0..k {
                        sum += *a.add(i * lda + kk) * *b.add(kk * ldb + j);
                    }
                    *out.add(i * ldc + j) = sum;
                    j += 1;
                }
                i += 1;
            }
        }
    };
}

// ---------------------------------------------------------------------------
// x86_64 instantiations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
define_small_matmul_regblocked_x86!(
    small_matmul_f32_avx2,
    f32,
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

#[cfg(target_arch = "x86_64")]
define_small_matmul_regblocked_x86!(
    small_matmul_f64_avx2,
    f64,
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

#[cfg(target_arch = "x86_64")]
define_small_matmul_regblocked_x86!(
    small_matmul_f32_avx512,
    f32,
    16,
    "avx512f",
    "fma",
    _mm512_loadu_ps,
    _mm512_storeu_ps,
    _mm512_set1_ps,
    _mm512_fmadd_ps,
    _mm512_setzero_ps,
    __m512
);

#[cfg(target_arch = "x86_64")]
define_small_matmul_regblocked_x86!(
    small_matmul_f64_avx512,
    f64,
    8,
    "avx512f",
    "fma",
    _mm512_loadu_pd,
    _mm512_storeu_pd,
    _mm512_set1_pd,
    _mm512_fmadd_pd,
    _mm512_setzero_pd,
    __m512d
);

#[cfg(target_arch = "x86_64")]
define_small_matmul_bias_regblocked_x86!(
    small_matmul_bias_f32_avx2,
    f32,
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

#[cfg(target_arch = "x86_64")]
define_small_matmul_bias_regblocked_x86!(
    small_matmul_bias_f64_avx2,
    f64,
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

#[cfg(target_arch = "x86_64")]
define_small_matmul_bias_regblocked_x86!(
    small_matmul_bias_f32_avx512,
    f32,
    16,
    "avx512f",
    "fma",
    _mm512_loadu_ps,
    _mm512_storeu_ps,
    _mm512_set1_ps,
    _mm512_fmadd_ps,
    _mm512_setzero_ps,
    __m512
);

#[cfg(target_arch = "x86_64")]
define_small_matmul_bias_regblocked_x86!(
    small_matmul_bias_f64_avx512,
    f64,
    8,
    "avx512f",
    "fma",
    _mm512_loadu_pd,
    _mm512_storeu_pd,
    _mm512_set1_pd,
    _mm512_fmadd_pd,
    _mm512_setzero_pd,
    __m512d
);

// ---------------------------------------------------------------------------
// aarch64 NEON register-blocked
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
macro_rules! define_small_matmul_regblocked_neon {
    ($name:ident, $ty:ty, $W:expr, $vld:ident, $vst:ident, $vdup:ident, $vfma:ident, $vec:ty) => {
        #[target_feature(enable = "neon")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            a: *const $ty,
            b: *const $ty,
            out: *mut $ty,
            m: usize,
            n: usize,
            k: usize,
            lda: usize,
            ldb: usize,
            ldc: usize,
        ) {
            use std::arch::aarch64::*;
            let mr = MR_SMALL;
            let mut i = 0;

            while i + mr <= m {
                let mut j = 0;
                while j + 2 * $W <= n {
                    let mut c00: $vec = $vdup(0.0 as $ty);
                    let mut c01: $vec = $vdup(0.0 as $ty);
                    let mut c10: $vec = $vdup(0.0 as $ty);
                    let mut c11: $vec = $vdup(0.0 as $ty);
                    let mut c20: $vec = $vdup(0.0 as $ty);
                    let mut c21: $vec = $vdup(0.0 as $ty);
                    let mut c30: $vec = $vdup(0.0 as $ty);
                    let mut c31: $vec = $vdup(0.0 as $ty);

                    for kk in 0..k {
                        let b0 = $vld(b.add(kk * ldb + j));
                        let b1 = $vld(b.add(kk * ldb + j + $W));

                        let a0 = $vdup(*a.add((i + 0) * lda + kk));
                        c00 = $vfma(c00, a0, b0);
                        c01 = $vfma(c01, a0, b1);

                        let a1 = $vdup(*a.add((i + 1) * lda + kk));
                        c10 = $vfma(c10, a1, b0);
                        c11 = $vfma(c11, a1, b1);

                        let a2 = $vdup(*a.add((i + 2) * lda + kk));
                        c20 = $vfma(c20, a2, b0);
                        c21 = $vfma(c21, a2, b1);

                        let a3 = $vdup(*a.add((i + 3) * lda + kk));
                        c30 = $vfma(c30, a3, b0);
                        c31 = $vfma(c31, a3, b1);
                    }

                    $vst(out.add((i + 0) * ldc + j), c00);
                    $vst(out.add((i + 0) * ldc + j + $W), c01);
                    $vst(out.add((i + 1) * ldc + j), c10);
                    $vst(out.add((i + 1) * ldc + j + $W), c11);
                    $vst(out.add((i + 2) * ldc + j), c20);
                    $vst(out.add((i + 2) * ldc + j + $W), c21);
                    $vst(out.add((i + 3) * ldc + j), c30);
                    $vst(out.add((i + 3) * ldc + j + $W), c31);
                    j += 2 * $W;
                }

                while j + $W <= n {
                    let mut c0: $vec = $vdup(0.0 as $ty);
                    let mut c1: $vec = $vdup(0.0 as $ty);
                    let mut c2: $vec = $vdup(0.0 as $ty);
                    let mut c3: $vec = $vdup(0.0 as $ty);
                    for kk in 0..k {
                        let bv = $vld(b.add(kk * ldb + j));
                        c0 = $vfma(c0, $vdup(*a.add((i + 0) * lda + kk)), bv);
                        c1 = $vfma(c1, $vdup(*a.add((i + 1) * lda + kk)), bv);
                        c2 = $vfma(c2, $vdup(*a.add((i + 2) * lda + kk)), bv);
                        c3 = $vfma(c3, $vdup(*a.add((i + 3) * lda + kk)), bv);
                    }
                    $vst(out.add((i + 0) * ldc + j), c0);
                    $vst(out.add((i + 1) * ldc + j), c1);
                    $vst(out.add((i + 2) * ldc + j), c2);
                    $vst(out.add((i + 3) * ldc + j), c3);
                    j += $W;
                }

                while j < n {
                    let mut s0: $ty = 0.0;
                    let mut s1: $ty = 0.0;
                    let mut s2: $ty = 0.0;
                    let mut s3: $ty = 0.0;
                    for kk in 0..k {
                        let bv = *b.add(kk * ldb + j);
                        s0 += *a.add((i + 0) * lda + kk) * bv;
                        s1 += *a.add((i + 1) * lda + kk) * bv;
                        s2 += *a.add((i + 2) * lda + kk) * bv;
                        s3 += *a.add((i + 3) * lda + kk) * bv;
                    }
                    *out.add((i + 0) * ldc + j) = s0;
                    *out.add((i + 1) * ldc + j) = s1;
                    *out.add((i + 2) * ldc + j) = s2;
                    *out.add((i + 3) * ldc + j) = s3;
                    j += 1;
                }

                i += mr;
            }

            while i < m {
                let mut j = 0;
                while j + $W <= n {
                    let mut acc: $vec = $vdup(0.0 as $ty);
                    for kk in 0..k {
                        acc = $vfma(acc, $vdup(*a.add(i * lda + kk)), $vld(b.add(kk * ldb + j)));
                    }
                    $vst(out.add(i * ldc + j), acc);
                    j += $W;
                }
                while j < n {
                    let mut sum: $ty = 0.0;
                    for kk in 0..k {
                        sum += *a.add(i * lda + kk) * *b.add(kk * ldb + j);
                    }
                    *out.add(i * ldc + j) = sum;
                    j += 1;
                }
                i += 1;
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
macro_rules! define_small_matmul_bias_regblocked_neon {
    ($name:ident, $ty:ty, $W:expr, $vld:ident, $vst:ident, $vdup:ident, $vfma:ident, $vec:ty) => {
        #[target_feature(enable = "neon")]
        #[allow(clippy::too_many_arguments)]
        pub unsafe fn $name(
            a: *const $ty,
            b: *const $ty,
            bias: *const $ty,
            out: *mut $ty,
            m: usize,
            n: usize,
            k: usize,
            lda: usize,
            ldb: usize,
            ldc: usize,
        ) {
            use std::arch::aarch64::*;
            let mr = MR_SMALL;
            let mut i = 0;

            while i + mr <= m {
                let mut j = 0;
                while j + 2 * $W <= n {
                    let bias0 = $vld(bias.add(j));
                    let bias1 = $vld(bias.add(j + $W));
                    let mut c00 = bias0;
                    let mut c01 = bias1;
                    let mut c10 = bias0;
                    let mut c11 = bias1;
                    let mut c20 = bias0;
                    let mut c21 = bias1;
                    let mut c30 = bias0;
                    let mut c31 = bias1;

                    for kk in 0..k {
                        let b0 = $vld(b.add(kk * ldb + j));
                        let b1 = $vld(b.add(kk * ldb + j + $W));
                        let a0 = $vdup(*a.add((i + 0) * lda + kk));
                        c00 = $vfma(c00, a0, b0);
                        c01 = $vfma(c01, a0, b1);
                        let a1 = $vdup(*a.add((i + 1) * lda + kk));
                        c10 = $vfma(c10, a1, b0);
                        c11 = $vfma(c11, a1, b1);
                        let a2 = $vdup(*a.add((i + 2) * lda + kk));
                        c20 = $vfma(c20, a2, b0);
                        c21 = $vfma(c21, a2, b1);
                        let a3 = $vdup(*a.add((i + 3) * lda + kk));
                        c30 = $vfma(c30, a3, b0);
                        c31 = $vfma(c31, a3, b1);
                    }

                    $vst(out.add((i + 0) * ldc + j), c00);
                    $vst(out.add((i + 0) * ldc + j + $W), c01);
                    $vst(out.add((i + 1) * ldc + j), c10);
                    $vst(out.add((i + 1) * ldc + j + $W), c11);
                    $vst(out.add((i + 2) * ldc + j), c20);
                    $vst(out.add((i + 2) * ldc + j + $W), c21);
                    $vst(out.add((i + 3) * ldc + j), c30);
                    $vst(out.add((i + 3) * ldc + j + $W), c31);
                    j += 2 * $W;
                }

                while j + $W <= n {
                    let biasv = $vld(bias.add(j));
                    let mut c0 = biasv;
                    let mut c1 = biasv;
                    let mut c2 = biasv;
                    let mut c3 = biasv;
                    for kk in 0..k {
                        let bv = $vld(b.add(kk * ldb + j));
                        c0 = $vfma(c0, $vdup(*a.add((i + 0) * lda + kk)), bv);
                        c1 = $vfma(c1, $vdup(*a.add((i + 1) * lda + kk)), bv);
                        c2 = $vfma(c2, $vdup(*a.add((i + 2) * lda + kk)), bv);
                        c3 = $vfma(c3, $vdup(*a.add((i + 3) * lda + kk)), bv);
                    }
                    $vst(out.add((i + 0) * ldc + j), c0);
                    $vst(out.add((i + 1) * ldc + j), c1);
                    $vst(out.add((i + 2) * ldc + j), c2);
                    $vst(out.add((i + 3) * ldc + j), c3);
                    j += $W;
                }

                while j < n {
                    let bval = *bias.add(j);
                    let mut s0 = bval;
                    let mut s1 = bval;
                    let mut s2 = bval;
                    let mut s3 = bval;
                    for kk in 0..k {
                        let bv = *b.add(kk * ldb + j);
                        s0 += *a.add((i + 0) * lda + kk) * bv;
                        s1 += *a.add((i + 1) * lda + kk) * bv;
                        s2 += *a.add((i + 2) * lda + kk) * bv;
                        s3 += *a.add((i + 3) * lda + kk) * bv;
                    }
                    *out.add((i + 0) * ldc + j) = s0;
                    *out.add((i + 1) * ldc + j) = s1;
                    *out.add((i + 2) * ldc + j) = s2;
                    *out.add((i + 3) * ldc + j) = s3;
                    j += 1;
                }

                i += mr;
            }

            while i < m {
                let mut j = 0;
                while j + $W <= n {
                    let mut acc = $vld(bias.add(j));
                    for kk in 0..k {
                        acc = $vfma(acc, $vdup(*a.add(i * lda + kk)), $vld(b.add(kk * ldb + j)));
                    }
                    $vst(out.add(i * ldc + j), acc);
                    j += $W;
                }
                while j < n {
                    let mut sum = *bias.add(j);
                    for kk in 0..k {
                        sum += *a.add(i * lda + kk) * *b.add(kk * ldb + j);
                    }
                    *out.add(i * ldc + j) = sum;
                    j += 1;
                }
                i += 1;
            }
        }
    };
}

// ---------------------------------------------------------------------------
// aarch64 instantiations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
define_small_matmul_regblocked_neon!(
    small_matmul_f32_neon,
    f32,
    4,
    vld1q_f32,
    vst1q_f32,
    vdupq_n_f32,
    vfmaq_f32,
    float32x4_t
);

#[cfg(target_arch = "aarch64")]
define_small_matmul_regblocked_neon!(
    small_matmul_f64_neon,
    f64,
    2,
    vld1q_f64,
    vst1q_f64,
    vdupq_n_f64,
    vfmaq_f64,
    float64x2_t
);

#[cfg(target_arch = "aarch64")]
define_small_matmul_bias_regblocked_neon!(
    small_matmul_bias_f32_neon,
    f32,
    4,
    vld1q_f32,
    vst1q_f32,
    vdupq_n_f32,
    vfmaq_f32,
    float32x4_t
);

#[cfg(target_arch = "aarch64")]
define_small_matmul_bias_regblocked_neon!(
    small_matmul_bias_f64_neon,
    f64,
    2,
    vld1q_f64,
    vst1q_f64,
    vdupq_n_f64,
    vfmaq_f64,
    float64x2_t
);
