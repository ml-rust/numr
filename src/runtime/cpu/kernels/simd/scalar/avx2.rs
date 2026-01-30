//! AVX2 scalar operation kernels
//!
//! Broadcasts scalar to vector and applies operation.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{scalar_scalar_f32, scalar_scalar_f64};
use crate::ops::BinaryOp;

const F32_LANES: usize = 8;
const F64_LANES: usize = 4;

/// AVX2 scalar operation for f32
#[target_feature(enable = "avx2")]
pub unsafe fn scalar_f32(op: BinaryOp, a: *const f32, scalar: f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    match op {
        BinaryOp::Add => scalar_add_f32(a, scalar, out, chunks),
        BinaryOp::Sub => scalar_sub_f32(a, scalar, out, chunks),
        BinaryOp::Mul => scalar_mul_f32(a, scalar, out, chunks),
        BinaryOp::Div => scalar_div_f32(a, scalar, out, chunks),
        BinaryOp::Max => scalar_max_f32(a, scalar, out, chunks),
        BinaryOp::Min => scalar_min_f32(a, scalar, out, chunks),
        _ => {
            scalar_scalar_f32(op, a, scalar, out, len);
            return;
        }
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        scalar_scalar_f32(op, a.add(offset), scalar, out.add(offset), remainder);
    }
}

/// AVX2 scalar operation for f64
#[target_feature(enable = "avx2")]
pub unsafe fn scalar_f64(op: BinaryOp, a: *const f64, scalar: f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    match op {
        BinaryOp::Add => scalar_add_f64(a, scalar, out, chunks),
        BinaryOp::Sub => scalar_sub_f64(a, scalar, out, chunks),
        BinaryOp::Mul => scalar_mul_f64(a, scalar, out, chunks),
        BinaryOp::Div => scalar_div_f64(a, scalar, out, chunks),
        BinaryOp::Max => scalar_max_f64(a, scalar, out, chunks),
        BinaryOp::Min => scalar_min_f64(a, scalar, out, chunks),
        _ => {
            scalar_scalar_f64(op, a, scalar, out, len);
            return;
        }
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        scalar_scalar_f64(op, a.add(offset), scalar, out.add(offset), remainder);
    }
}

// ============================================================================
// f32 kernels
// ============================================================================

#[target_feature(enable = "avx2")]
unsafe fn scalar_add_f32(a: *const f32, scalar: f32, out: *mut f32, chunks: usize) {
    let vs = _mm256_set1_ps(scalar);
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = _mm256_add_ps(va, vs);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_sub_f32(a: *const f32, scalar: f32, out: *mut f32, chunks: usize) {
    let vs = _mm256_set1_ps(scalar);
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = _mm256_sub_ps(va, vs);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_mul_f32(a: *const f32, scalar: f32, out: *mut f32, chunks: usize) {
    let vs = _mm256_set1_ps(scalar);
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_div_f32(a: *const f32, scalar: f32, out: *mut f32, chunks: usize) {
    let vs = _mm256_set1_ps(scalar);
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = _mm256_div_ps(va, vs);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_max_f32(a: *const f32, scalar: f32, out: *mut f32, chunks: usize) {
    let vs = _mm256_set1_ps(scalar);
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = _mm256_max_ps(va, vs);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_min_f32(a: *const f32, scalar: f32, out: *mut f32, chunks: usize) {
    let vs = _mm256_set1_ps(scalar);
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = _mm256_min_ps(va, vs);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

// ============================================================================
// f64 kernels
// ============================================================================

#[target_feature(enable = "avx2")]
unsafe fn scalar_add_f64(a: *const f64, scalar: f64, out: *mut f64, chunks: usize) {
    let vs = _mm256_set1_pd(scalar);
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = _mm256_add_pd(va, vs);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_sub_f64(a: *const f64, scalar: f64, out: *mut f64, chunks: usize) {
    let vs = _mm256_set1_pd(scalar);
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = _mm256_sub_pd(va, vs);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_mul_f64(a: *const f64, scalar: f64, out: *mut f64, chunks: usize) {
    let vs = _mm256_set1_pd(scalar);
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = _mm256_mul_pd(va, vs);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_div_f64(a: *const f64, scalar: f64, out: *mut f64, chunks: usize) {
    let vs = _mm256_set1_pd(scalar);
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = _mm256_div_pd(va, vs);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_max_f64(a: *const f64, scalar: f64, out: *mut f64, chunks: usize) {
    let vs = _mm256_set1_pd(scalar);
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = _mm256_max_pd(va, vs);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn scalar_min_f64(a: *const f64, scalar: f64, out: *mut f64, chunks: usize) {
    let vs = _mm256_set1_pd(scalar);
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = _mm256_min_pd(va, vs);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}
