//! AVX-512 comparison kernels
//!
//! Uses mask-based comparison intrinsics and blend operations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{compare_scalar_f32, compare_scalar_f64};
use crate::ops::CompareOp;

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

// AVX-512 comparison predicates
const CMP_EQ_OQ: i32 = 0; // Equal (ordered, quiet)
const CMP_LT_OS: i32 = 1; // Less than (ordered, signaling)
const CMP_LE_OS: i32 = 2; // Less than or equal (ordered, signaling)
const CMP_NEQ_UQ: i32 = 4; // Not equal (unordered, quiet)
const CMP_NLT_US: i32 = 5; // Not less than = GE (unordered, signaling)
const CMP_NLE_US: i32 = 6; // Not less than or equal = GT (unordered, signaling)

/// AVX-512 comparison for f32
#[target_feature(enable = "avx512f")]
pub unsafe fn compare_f32(op: CompareOp, a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let ones = _mm512_set1_ps(1.0);
    let zeros = _mm512_setzero_ps();

    for c in 0..chunks {
        let offset = c * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));

        let mask = match op {
            CompareOp::Eq => _mm512_cmp_ps_mask::<{ CMP_EQ_OQ }>(va, vb),
            CompareOp::Ne => _mm512_cmp_ps_mask::<{ CMP_NEQ_UQ }>(va, vb),
            CompareOp::Lt => _mm512_cmp_ps_mask::<{ CMP_LT_OS }>(va, vb),
            CompareOp::Le => _mm512_cmp_ps_mask::<{ CMP_LE_OS }>(va, vb),
            CompareOp::Gt => _mm512_cmp_ps_mask::<{ CMP_NLE_US }>(va, vb),
            CompareOp::Ge => _mm512_cmp_ps_mask::<{ CMP_NLT_US }>(va, vb),
        };

        // Blend: mask=1 → ones, mask=0 → zeros
        let result = _mm512_mask_blend_ps(mask, zeros, ones);
        _mm512_storeu_ps(out.add(offset), result);
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

/// AVX-512 comparison for f64
#[target_feature(enable = "avx512f")]
pub unsafe fn compare_f64(op: CompareOp, a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let ones = _mm512_set1_pd(1.0);
    let zeros = _mm512_setzero_pd();

    for c in 0..chunks {
        let offset = c * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vb = _mm512_loadu_pd(b.add(offset));

        let mask = match op {
            CompareOp::Eq => _mm512_cmp_pd_mask::<{ CMP_EQ_OQ }>(va, vb),
            CompareOp::Ne => _mm512_cmp_pd_mask::<{ CMP_NEQ_UQ }>(va, vb),
            CompareOp::Lt => _mm512_cmp_pd_mask::<{ CMP_LT_OS }>(va, vb),
            CompareOp::Le => _mm512_cmp_pd_mask::<{ CMP_LE_OS }>(va, vb),
            CompareOp::Gt => _mm512_cmp_pd_mask::<{ CMP_NLE_US }>(va, vb),
            CompareOp::Ge => _mm512_cmp_pd_mask::<{ CMP_NLT_US }>(va, vb),
        };

        let result = _mm512_mask_blend_pd(mask, zeros, ones);
        _mm512_storeu_pd(out.add(offset), result);
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
