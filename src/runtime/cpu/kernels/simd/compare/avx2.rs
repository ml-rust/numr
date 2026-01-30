//! AVX2 comparison kernels
//!
//! Uses vector comparison intrinsics and blend operations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{compare_scalar_f32, compare_scalar_f64};
use crate::ops::CompareOp;

const F32_LANES: usize = 8;
const F64_LANES: usize = 4;

// AVX comparison predicates (same as AVX-512)
const CMP_EQ_OQ: i32 = 0; // Equal (ordered, quiet)
const CMP_LT_OS: i32 = 1; // Less than (ordered, signaling)
const CMP_LE_OS: i32 = 2; // Less than or equal (ordered, signaling)
const CMP_NEQ_UQ: i32 = 4; // Not equal (unordered, quiet)
const CMP_NLT_US: i32 = 5; // Not less than = GE (unordered, signaling)
const CMP_NLE_US: i32 = 6; // Not less than or equal = GT (unordered, signaling)

/// AVX2 comparison for f32
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn compare_f32(op: CompareOp, a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let ones = _mm256_set1_ps(1.0);
    let zeros = _mm256_setzero_ps();

    for c in 0..chunks {
        let offset = c * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vb = _mm256_loadu_ps(b.add(offset));

        // _mm256_cmp_ps returns all 1s (0xFFFFFFFF) for true, all 0s for false
        let mask = match op {
            CompareOp::Eq => _mm256_cmp_ps::<{ CMP_EQ_OQ }>(va, vb),
            CompareOp::Ne => _mm256_cmp_ps::<{ CMP_NEQ_UQ }>(va, vb),
            CompareOp::Lt => _mm256_cmp_ps::<{ CMP_LT_OS }>(va, vb),
            CompareOp::Le => _mm256_cmp_ps::<{ CMP_LE_OS }>(va, vb),
            CompareOp::Gt => _mm256_cmp_ps::<{ CMP_NLE_US }>(va, vb),
            CompareOp::Ge => _mm256_cmp_ps::<{ CMP_NLT_US }>(va, vb),
        };

        // Blend: mask bit set → ones, mask bit clear → zeros
        let result = _mm256_blendv_ps(zeros, ones, mask);
        _mm256_storeu_ps(out.add(offset), result);
    }

    // Scalar tail
    let processed = chunks * F32_LANES;
    if processed < len {
        compare_scalar_f32(
            op,
            a.add(processed),
            b.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

/// AVX2 comparison for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn compare_f64(op: CompareOp, a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let ones = _mm256_set1_pd(1.0);
    let zeros = _mm256_setzero_pd();

    for c in 0..chunks {
        let offset = c * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vb = _mm256_loadu_pd(b.add(offset));

        let mask = match op {
            CompareOp::Eq => _mm256_cmp_pd::<{ CMP_EQ_OQ }>(va, vb),
            CompareOp::Ne => _mm256_cmp_pd::<{ CMP_NEQ_UQ }>(va, vb),
            CompareOp::Lt => _mm256_cmp_pd::<{ CMP_LT_OS }>(va, vb),
            CompareOp::Le => _mm256_cmp_pd::<{ CMP_LE_OS }>(va, vb),
            CompareOp::Gt => _mm256_cmp_pd::<{ CMP_NLE_US }>(va, vb),
            CompareOp::Ge => _mm256_cmp_pd::<{ CMP_NLT_US }>(va, vb),
        };

        let result = _mm256_blendv_pd(zeros, ones, mask);
        _mm256_storeu_pd(out.add(offset), result);
    }

    // Scalar tail
    let processed = chunks * F64_LANES;
    if processed < len {
        compare_scalar_f64(
            op,
            a.add(processed),
            b.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

// Suppress unused warnings for scalar fallback imports
const _: () = {
    let _ = compare_scalar_f32 as unsafe fn(CompareOp, *const f32, *const f32, *mut f32, usize);
    let _ = compare_scalar_f64 as unsafe fn(CompareOp, *const f64, *const f64, *mut f64, usize);
};
