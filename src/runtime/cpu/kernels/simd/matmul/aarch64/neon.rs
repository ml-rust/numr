//! NEON matmul microkernels for ARM64
//!
//! Provides vectorized matrix multiplication microkernels using 128-bit NEON registers.
//!
//! # Microkernel Dimensions
//!
//! - f32: 6×4 (6 rows × 4 columns = 24 elements per microkernel invocation)
//! - f64: 6×2 (6 rows × 2 columns = 12 elements per microkernel invocation)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Matmul microkernel 6x4 for f32: C[0:6, 0:4] += A[0:6, 0:K] @ B[0:K, 0:4]
///
/// When `first_k` is true, accumulators start from zero (beta=0).
/// When false, they load from C and accumulate (beta=1).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn microkernel_6x4_f32(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    ldc: usize,
    first_k: bool,
) {
    let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5);

    if first_k {
        c0 = vdupq_n_f32(0.0);
        c1 = vdupq_n_f32(0.0);
        c2 = vdupq_n_f32(0.0);
        c3 = vdupq_n_f32(0.0);
        c4 = vdupq_n_f32(0.0);
        c5 = vdupq_n_f32(0.0);
    } else {
        c0 = vld1q_f32(c);
        c1 = vld1q_f32(c.add(ldc));
        c2 = vld1q_f32(c.add(ldc * 2));
        c3 = vld1q_f32(c.add(ldc * 3));
        c4 = vld1q_f32(c.add(ldc * 4));
        c5 = vld1q_f32(c.add(ldc * 5));
    }

    for kk in 0..k {
        let b_row = vld1q_f32(b.add(kk * 4));
        let a_base = a.add(kk * 6);

        let a0 = vld1q_dup_f32(a_base);
        c0 = vfmaq_f32(c0, a0, b_row);

        let a1 = vld1q_dup_f32(a_base.add(1));
        c1 = vfmaq_f32(c1, a1, b_row);

        let a2 = vld1q_dup_f32(a_base.add(2));
        c2 = vfmaq_f32(c2, a2, b_row);

        let a3 = vld1q_dup_f32(a_base.add(3));
        c3 = vfmaq_f32(c3, a3, b_row);

        let a4 = vld1q_dup_f32(a_base.add(4));
        c4 = vfmaq_f32(c4, a4, b_row);

        let a5 = vld1q_dup_f32(a_base.add(5));
        c5 = vfmaq_f32(c5, a5, b_row);
    }

    vst1q_f32(c, c0);
    vst1q_f32(c.add(ldc), c1);
    vst1q_f32(c.add(ldc * 2), c2);
    vst1q_f32(c.add(ldc * 3), c3);
    vst1q_f32(c.add(ldc * 4), c4);
    vst1q_f32(c.add(ldc * 5), c5);
}

/// Matmul microkernel 6x2 for f64: C[0:6, 0:2] += A[0:6, 0:K] @ B[0:K, 0:2]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn microkernel_6x2_f64(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    k: usize,
    ldc: usize,
    first_k: bool,
) {
    let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5);

    if first_k {
        c0 = vdupq_n_f64(0.0);
        c1 = vdupq_n_f64(0.0);
        c2 = vdupq_n_f64(0.0);
        c3 = vdupq_n_f64(0.0);
        c4 = vdupq_n_f64(0.0);
        c5 = vdupq_n_f64(0.0);
    } else {
        c0 = vld1q_f64(c);
        c1 = vld1q_f64(c.add(ldc));
        c2 = vld1q_f64(c.add(ldc * 2));
        c3 = vld1q_f64(c.add(ldc * 3));
        c4 = vld1q_f64(c.add(ldc * 4));
        c5 = vld1q_f64(c.add(ldc * 5));
    }

    for kk in 0..k {
        let b_row = vld1q_f64(b.add(kk * 2));
        let a_base = a.add(kk * 6);

        let a0 = vld1q_dup_f64(a_base);
        c0 = vfmaq_f64(c0, a0, b_row);

        let a1 = vld1q_dup_f64(a_base.add(1));
        c1 = vfmaq_f64(c1, a1, b_row);

        let a2 = vld1q_dup_f64(a_base.add(2));
        c2 = vfmaq_f64(c2, a2, b_row);

        let a3 = vld1q_dup_f64(a_base.add(3));
        c3 = vfmaq_f64(c3, a3, b_row);

        let a4 = vld1q_dup_f64(a_base.add(4));
        c4 = vfmaq_f64(c4, a4, b_row);

        let a5 = vld1q_dup_f64(a_base.add(5));
        c5 = vfmaq_f64(c5, a5, b_row);
    }

    vst1q_f64(c, c0);
    vst1q_f64(c.add(ldc), c1);
    vst1q_f64(c.add(ldc * 2), c2);
    vst1q_f64(c.add(ldc * 3), c3);
    vst1q_f64(c.add(ldc * 4), c4);
    vst1q_f64(c.add(ldc * 5), c5);
}
