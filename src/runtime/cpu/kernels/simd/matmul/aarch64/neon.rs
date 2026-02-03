//! NEON matmul microkernels for ARM64
//!
//! Provides vectorized matrix multiplication microkernels using 128-bit NEON registers.
//!
//! # Microkernel Dimensions
//!
//! - f32: 6×4 (6 rows × 4 columns = 24 elements per microkernel invocation)
//! - f64: 6×2 (6 rows × 2 columns = 12 elements per microkernel invocation)
//!
//! # Register Usage (f32 6x4)
//!
//! - v0-v5: C accumulators (6 rows × 4 columns)
//! - v6: A broadcast register
//! - v7: B load register
//!
//! # Algorithm
//!
//! ```text
//! for kk in 0..k:
//!     b_row = load B[kk, 0:NR]
//!     for i in 0..MR:
//!         a_i = broadcast A[i, kk]
//!         C[i] += a_i * b_row  (FMA)
//! store C accumulators
//! ```

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Matmul microkernel 6x4 for f32: C[0:6, 0:4] += A[0:6, 0:K] @ B[0:K, 0:4]
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a` must point to `k * 6` valid f32 elements (packed row panel)
/// - `b` must point to `k * 4` valid f32 elements (packed row panel)
/// - `c` must point to start of output with stride `ldc`
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn microkernel_6x4_f32(a: *const f32, b: *const f32, c: *mut f32, k: usize, ldc: usize) {
    // Load C accumulators (6 rows, 4 columns each)
    let mut c0 = vld1q_f32(c);
    let mut c1 = vld1q_f32(c.add(ldc));
    let mut c2 = vld1q_f32(c.add(ldc * 2));
    let mut c3 = vld1q_f32(c.add(ldc * 3));
    let mut c4 = vld1q_f32(c.add(ldc * 4));
    let mut c5 = vld1q_f32(c.add(ldc * 5));

    for kk in 0..k {
        // Load B row (4 elements)
        let b_row = vld1q_f32(b.add(kk * 4));
        let a_base = a.add(kk * 6);

        // Row 0: broadcast A[0,kk], FMA with B row
        let a0 = vld1q_dup_f32(a_base);
        c0 = vfmaq_f32(c0, a0, b_row);

        // Row 1
        let a1 = vld1q_dup_f32(a_base.add(1));
        c1 = vfmaq_f32(c1, a1, b_row);

        // Row 2
        let a2 = vld1q_dup_f32(a_base.add(2));
        c2 = vfmaq_f32(c2, a2, b_row);

        // Row 3
        let a3 = vld1q_dup_f32(a_base.add(3));
        c3 = vfmaq_f32(c3, a3, b_row);

        // Row 4
        let a4 = vld1q_dup_f32(a_base.add(4));
        c4 = vfmaq_f32(c4, a4, b_row);

        // Row 5
        let a5 = vld1q_dup_f32(a_base.add(5));
        c5 = vfmaq_f32(c5, a5, b_row);
    }

    // Store C accumulators
    vst1q_f32(c, c0);
    vst1q_f32(c.add(ldc), c1);
    vst1q_f32(c.add(ldc * 2), c2);
    vst1q_f32(c.add(ldc * 3), c3);
    vst1q_f32(c.add(ldc * 4), c4);
    vst1q_f32(c.add(ldc * 5), c5);
}

/// Matmul microkernel 6x2 for f64: C[0:6, 0:2] += A[0:6, 0:K] @ B[0:K, 0:2]
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a` must point to `k * 6` valid f64 elements (packed row panel)
/// - `b` must point to `k * 2` valid f64 elements (packed row panel)
/// - `c` must point to start of output with stride `ldc`
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn microkernel_6x2_f64(a: *const f64, b: *const f64, c: *mut f64, k: usize, ldc: usize) {
    // Load C accumulators (6 rows, 2 columns each)
    let mut c0 = vld1q_f64(c);
    let mut c1 = vld1q_f64(c.add(ldc));
    let mut c2 = vld1q_f64(c.add(ldc * 2));
    let mut c3 = vld1q_f64(c.add(ldc * 3));
    let mut c4 = vld1q_f64(c.add(ldc * 4));
    let mut c5 = vld1q_f64(c.add(ldc * 5));

    for kk in 0..k {
        // Load B row (2 elements)
        let b_row = vld1q_f64(b.add(kk * 2));
        let a_base = a.add(kk * 6);

        // Row 0
        let a0 = vld1q_dup_f64(a_base);
        c0 = vfmaq_f64(c0, a0, b_row);

        // Row 1
        let a1 = vld1q_dup_f64(a_base.add(1));
        c1 = vfmaq_f64(c1, a1, b_row);

        // Row 2
        let a2 = vld1q_dup_f64(a_base.add(2));
        c2 = vfmaq_f64(c2, a2, b_row);

        // Row 3
        let a3 = vld1q_dup_f64(a_base.add(3));
        c3 = vfmaq_f64(c3, a3, b_row);

        // Row 4
        let a4 = vld1q_dup_f64(a_base.add(4));
        c4 = vfmaq_f64(c4, a4, b_row);

        // Row 5
        let a5 = vld1q_dup_f64(a_base.add(5));
        c5 = vfmaq_f64(c5, a5, b_row);
    }

    // Store C accumulators
    vst1q_f64(c, c0);
    vst1q_f64(c.add(ldc), c1);
    vst1q_f64(c.add(ldc * 2), c2);
    vst1q_f64(c.add(ldc * 3), c3);
    vst1q_f64(c.add(ldc * 4), c4);
    vst1q_f64(c.add(ldc * 5), c5);
}
