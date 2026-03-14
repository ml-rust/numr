//! GEMV-BT kernel: C[M,N] = A[M,K] @ B^T where B is stored as [N,K]
//!
//! When a weight matrix W[N,K] is transposed to get W^T[K,N], the result has
//! shape [K,N] and strides [1,K] — it's a view into the original [N,K] data.
//! Rather than copying to make it contiguous (which allocates K*N elements),
//! we can compute the matmul directly: each output C[m,n] = dot(A[m,:], B[n,:])
//! where both A[m,:] and B[n,:] are contiguous K-element vectors.
//!
//! For decode (M=1), this eliminates:
//! - The contiguous copy of the entire weight matrix (e.g. 500MB for lm_head)
//! - The full B→f32 conversion buffer allocation (another 1GB for BF16)
//! - The overhead of the tiled GEMM algorithm for a single row

use super::super::SimdLevel;

/// GEMV-BT for f32: C[M,N] = A[M,K] @ B^T, B stored [N,K] row-major
///
/// # Safety
/// - `a` must point to M*K contiguous f32 elements (row-major, stride=K)
/// - `b` must point to N*K contiguous f32 elements (row-major, stride=K)
/// - `out` must point to M*N writable f32 elements (row-major, stride=ldc)
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_bt_f32(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    level: SimdLevel,
) {
    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => gemv_bt_f32_avx512(a, b, out, m, n, k, ldc),
        SimdLevel::Avx2Fma => gemv_bt_f32_avx2(a, b, out, m, n, k, ldc),
        _ => gemv_bt_f32_scalar(a, b, out, m, n, k, ldc),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => gemv_bt_f32_neon(a, b, out, m, n, k, ldc),
        _ => gemv_bt_f32_scalar(a, b, out, m, n, k, ldc),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = level;
        gemv_bt_f32_scalar(a, b, out, m, n, k, ldc);
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn gemv_bt_f32_scalar(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    for row in 0..m {
        let a_row = a.add(row * k);
        let out_row = out.add(row * ldc);
        for col in 0..n {
            let b_row = b.add(col * k);
            let mut sum = 0.0f32;
            for i in 0..k {
                sum += *a_row.add(i) * *b_row.add(i);
            }
            *out_row.add(col) = sum;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_bt_f32_avx2(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    use std::arch::x86_64::*;

    for row in 0..m {
        let a_row = a.add(row * k);
        let out_row = out.add(row * ldc);

        // Process 4 output columns at a time for better ILP
        let mut col = 0usize;
        while col + 4 <= n {
            let b0 = b.add(col * k);
            let b1 = b.add((col + 1) * k);
            let b2 = b.add((col + 2) * k);
            let b3 = b.add((col + 3) * k);

            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();

            let mut i = 0usize;
            while i + 8 <= k {
                let av = _mm256_loadu_ps(a_row.add(i));
                acc0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b0.add(i)), acc0);
                acc1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b1.add(i)), acc1);
                acc2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b2.add(i)), acc2);
                acc3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b3.add(i)), acc3);
                i += 8;
            }

            let mut s0 = hsum_avx2(acc0);
            let mut s1 = hsum_avx2(acc1);
            let mut s2 = hsum_avx2(acc2);
            let mut s3 = hsum_avx2(acc3);

            // Scalar tail
            while i < k {
                let av = *a_row.add(i);
                s0 += av * *b0.add(i);
                s1 += av * *b1.add(i);
                s2 += av * *b2.add(i);
                s3 += av * *b3.add(i);
                i += 1;
            }

            *out_row.add(col) = s0;
            *out_row.add(col + 1) = s1;
            *out_row.add(col + 2) = s2;
            *out_row.add(col + 3) = s3;
            col += 4;
        }

        // Remaining columns
        while col < n {
            let b_row = b.add(col * k);
            let mut acc = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 8 <= k {
                let av = _mm256_loadu_ps(a_row.add(i));
                acc = _mm256_fmadd_ps(av, _mm256_loadu_ps(b_row.add(i)), acc);
                i += 8;
            }
            let mut s = hsum_avx2(acc);
            while i < k {
                s += *a_row.add(i) * *b_row.add(i);
                i += 1;
            }
            *out_row.add(col) = s;
            col += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn hsum_avx2(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // [a0+a4, a1+a5, a2+a6, a3+a7] as 128-bit
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    // [s0+s2, s1+s3, ...]
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_bt_f32_avx512(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    use std::arch::x86_64::*;

    for row in 0..m {
        let a_row = a.add(row * k);
        let out_row = out.add(row * ldc);

        // Process 4 output columns at a time
        let mut col = 0usize;
        while col + 4 <= n {
            let b0 = b.add(col * k);
            let b1 = b.add((col + 1) * k);
            let b2 = b.add((col + 2) * k);
            let b3 = b.add((col + 3) * k);

            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();

            let mut i = 0usize;
            while i + 16 <= k {
                let av = _mm512_loadu_ps(a_row.add(i));
                acc0 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b0.add(i)), acc0);
                acc1 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b1.add(i)), acc1);
                acc2 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b2.add(i)), acc2);
                acc3 = _mm512_fmadd_ps(av, _mm512_loadu_ps(b3.add(i)), acc3);
                i += 16;
            }

            let mut s0 = _mm512_reduce_add_ps(acc0);
            let mut s1 = _mm512_reduce_add_ps(acc1);
            let mut s2 = _mm512_reduce_add_ps(acc2);
            let mut s3 = _mm512_reduce_add_ps(acc3);

            while i < k {
                let av = *a_row.add(i);
                s0 += av * *b0.add(i);
                s1 += av * *b1.add(i);
                s2 += av * *b2.add(i);
                s3 += av * *b3.add(i);
                i += 1;
            }

            *out_row.add(col) = s0;
            *out_row.add(col + 1) = s1;
            *out_row.add(col + 2) = s2;
            *out_row.add(col + 3) = s3;
            col += 4;
        }

        while col < n {
            let b_row = b.add(col * k);
            let mut acc = _mm512_setzero_ps();
            let mut i = 0usize;
            while i + 16 <= k {
                let av = _mm512_loadu_ps(a_row.add(i));
                acc = _mm512_fmadd_ps(av, _mm512_loadu_ps(b_row.add(i)), acc);
                i += 16;
            }
            let mut s = _mm512_reduce_add_ps(acc);
            while i < k {
                s += *a_row.add(i) * *b_row.add(i);
                i += 1;
            }
            *out_row.add(col) = s;
            col += 1;
        }
    }
}

/// GEMV-BT for f64: C[M,N] = A[M,K] @ B^T, B stored [N,K] row-major
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_bt_f64(
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    level: SimdLevel,
) {
    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => gemv_bt_f64_avx512(a, b, out, m, n, k, ldc),
        SimdLevel::Avx2Fma => gemv_bt_f64_avx2(a, b, out, m, n, k, ldc),
        _ => gemv_bt_f64_scalar(a, b, out, m, n, k, ldc),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => gemv_bt_f64_neon(a, b, out, m, n, k, ldc),
        _ => gemv_bt_f64_scalar(a, b, out, m, n, k, ldc),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = level;
        gemv_bt_f64_scalar(a, b, out, m, n, k, ldc);
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn gemv_bt_f64_scalar(
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    for row in 0..m {
        let a_row = a.add(row * k);
        let out_row = out.add(row * ldc);
        for col in 0..n {
            let b_row = b.add(col * k);
            let mut sum = 0.0f64;
            for i in 0..k {
                sum += *a_row.add(i) * *b_row.add(i);
            }
            *out_row.add(col) = sum;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_bt_f64_avx2(
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    use std::arch::x86_64::*;

    for row in 0..m {
        let a_row = a.add(row * k);
        let out_row = out.add(row * ldc);

        for col in 0..n {
            let b_row = b.add(col * k);
            let mut acc0 = _mm256_setzero_pd();
            let mut acc1 = _mm256_setzero_pd();

            let mut i = 0usize;
            while i + 8 <= k {
                acc0 = _mm256_fmadd_pd(
                    _mm256_loadu_pd(a_row.add(i)),
                    _mm256_loadu_pd(b_row.add(i)),
                    acc0,
                );
                acc1 = _mm256_fmadd_pd(
                    _mm256_loadu_pd(a_row.add(i + 4)),
                    _mm256_loadu_pd(b_row.add(i + 4)),
                    acc1,
                );
                i += 8;
            }
            let mut acc = _mm256_add_pd(acc0, acc1);

            while i + 4 <= k {
                acc = _mm256_fmadd_pd(
                    _mm256_loadu_pd(a_row.add(i)),
                    _mm256_loadu_pd(b_row.add(i)),
                    acc,
                );
                i += 4;
            }

            let mut s = hsum_avx2_f64(acc);
            while i < k {
                s += *a_row.add(i) * *b_row.add(i);
                i += 1;
            }
            *out_row.add(col) = s;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_avx2_f64(v: std::arch::x86_64::__m256d) -> f64 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_pd(v, 1);
    let lo = _mm256_castpd256_pd128(v);
    let sum128 = _mm_add_pd(lo, hi);
    let hi64 = _mm_unpackhi_pd(sum128, sum128);
    let sum = _mm_add_sd(sum128, hi64);
    _mm_cvtsd_f64(sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_bt_f64_avx512(
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    use std::arch::x86_64::*;

    for row in 0..m {
        let a_row = a.add(row * k);
        let out_row = out.add(row * ldc);

        for col in 0..n {
            let b_row = b.add(col * k);
            let mut acc = _mm512_setzero_pd();
            let mut i = 0usize;
            while i + 8 <= k {
                let av = _mm512_loadu_pd(a_row.add(i));
                acc = _mm512_fmadd_pd(av, _mm512_loadu_pd(b_row.add(i)), acc);
                i += 8;
            }
            let mut s = _mm512_reduce_add_pd(acc);
            while i < k {
                s += *a_row.add(i) * *b_row.add(i);
                i += 1;
            }
            *out_row.add(col) = s;
        }
    }
}

// ============================================================================
// NEON implementations (aarch64)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_bt_f32_neon(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    use std::arch::aarch64::*;

    for row in 0..m {
        let a_row = a.add(row * k);
        let out_row = out.add(row * ldc);

        // Process 4 output columns at a time
        let mut col = 0usize;
        while col + 4 <= n {
            let b0 = b.add(col * k);
            let b1 = b.add((col + 1) * k);
            let b2 = b.add((col + 2) * k);
            let b3 = b.add((col + 3) * k);

            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            let mut i = 0usize;
            while i + 4 <= k {
                let av = vld1q_f32(a_row.add(i));
                acc0 = vfmaq_f32(acc0, av, vld1q_f32(b0.add(i)));
                acc1 = vfmaq_f32(acc1, av, vld1q_f32(b1.add(i)));
                acc2 = vfmaq_f32(acc2, av, vld1q_f32(b2.add(i)));
                acc3 = vfmaq_f32(acc3, av, vld1q_f32(b3.add(i)));
                i += 4;
            }

            let mut s0 = vaddvq_f32(acc0);
            let mut s1 = vaddvq_f32(acc1);
            let mut s2 = vaddvq_f32(acc2);
            let mut s3 = vaddvq_f32(acc3);

            while i < k {
                let av = *a_row.add(i);
                s0 += av * *b0.add(i);
                s1 += av * *b1.add(i);
                s2 += av * *b2.add(i);
                s3 += av * *b3.add(i);
                i += 1;
            }

            *out_row.add(col) = s0;
            *out_row.add(col + 1) = s1;
            *out_row.add(col + 2) = s2;
            *out_row.add(col + 3) = s3;
            col += 4;
        }

        while col < n {
            let b_row = b.add(col * k);
            let mut acc = vdupq_n_f32(0.0);
            let mut i = 0usize;
            while i + 4 <= k {
                acc = vfmaq_f32(acc, vld1q_f32(a_row.add(i)), vld1q_f32(b_row.add(i)));
                i += 4;
            }
            let mut s = vaddvq_f32(acc);
            while i < k {
                s += *a_row.add(i) * *b_row.add(i);
                i += 1;
            }
            *out_row.add(col) = s;
            col += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_bt_f64_neon(
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    use std::arch::aarch64::*;

    for row in 0..m {
        let a_row = a.add(row * k);
        let out_row = out.add(row * ldc);

        for col in 0..n {
            let b_row = b.add(col * k);
            let mut acc0 = vdupq_n_f64(0.0);
            let mut acc1 = vdupq_n_f64(0.0);

            let mut i = 0usize;
            while i + 4 <= k {
                acc0 = vfmaq_f64(acc0, vld1q_f64(a_row.add(i)), vld1q_f64(b_row.add(i)));
                acc1 = vfmaq_f64(
                    acc1,
                    vld1q_f64(a_row.add(i + 2)),
                    vld1q_f64(b_row.add(i + 2)),
                );
                i += 4;
            }
            let mut acc = vaddq_f64(acc0, acc1);

            while i + 2 <= k {
                acc = vfmaq_f64(acc, vld1q_f64(a_row.add(i)), vld1q_f64(b_row.add(i)));
                i += 2;
            }

            let mut s = vaddvq_f64(acc);
            while i < k {
                s += *a_row.add(i) * *b_row.add(i);
                i += 1;
            }
            *out_row.add(col) = s;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_gemv_bt(a: &[f32], b_nk: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * b_nk[j * k + kk];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    #[test]
    fn test_gemv_bt_f32_m1() {
        let (m, n, k) = (1, 64, 128);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1).collect();
        let b: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32) * 0.1).collect();
        let mut c = vec![0.0f32; m * n];
        let expected = reference_gemv_bt(&a, &b, m, n, k);

        let level = super::super::super::detect_simd();
        unsafe { gemv_bt_f32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, n, level) };

        let max_diff = c
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "max_diff={max_diff}");
    }

    #[test]
    fn test_gemv_bt_f32_m4() {
        let (m, n, k) = (4, 53, 97);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.3).collect();
        let b: Vec<f32> = (0..n * k).map(|i| ((i % 11) as f32) * 0.2).collect();
        let mut c = vec![0.0f32; m * n];
        let expected = reference_gemv_bt(&a, &b, m, n, k);

        let level = super::super::super::detect_simd();
        unsafe { gemv_bt_f32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, n, level) };

        let max_diff = c
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "max_diff={max_diff}");
    }

    #[test]
    fn test_gemv_bt_f64_m1() {
        let (m, n, k) = (1, 64, 128);
        let a: Vec<f64> = (0..m * k).map(|i| ((i % 17) as f64) * 0.1).collect();
        let b_nk: Vec<f64> = (0..n * k).map(|i| ((i % 13) as f64) * 0.1).collect();
        let mut c = vec![0.0f64; m * n];

        // Reference
        let mut expected = vec![0.0f64; m * n];
        for j in 0..n {
            let mut sum = 0.0f64;
            for kk in 0..k {
                sum += a[kk] * b_nk[j * k + kk];
            }
            expected[j] = sum;
        }

        let level = super::super::super::detect_simd();
        unsafe { gemv_bt_f64(a.as_ptr(), b_nk.as_ptr(), c.as_mut_ptr(), m, n, k, n, level) };

        let max_diff = c
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < 1e-10, "max_diff={max_diff}");
    }
}
