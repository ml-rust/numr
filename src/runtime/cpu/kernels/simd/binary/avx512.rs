//! AVX-512 binary operation kernels
//!
//! Processes 16 f32s or 8 f64s per iteration using 512-bit vectors.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{binary_scalar_f32, binary_scalar_f64};
use crate::ops::BinaryOp;

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

/// AVX-512 binary operation for f32
///
/// # Safety
/// - CPU must support AVX-512F
/// - All pointers must be valid for `len` elements
#[target_feature(enable = "avx512f")]
pub unsafe fn binary_f32(op: BinaryOp, a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    match op {
        BinaryOp::Add => binary_add_f32_avx512(a, b, out, chunks),
        BinaryOp::Sub => binary_sub_f32_avx512(a, b, out, chunks),
        BinaryOp::Mul => binary_mul_f32_avx512(a, b, out, chunks),
        BinaryOp::Div => binary_div_f32_avx512(a, b, out, chunks),
        BinaryOp::Max => binary_max_f32_avx512(a, b, out, chunks),
        BinaryOp::Min => binary_min_f32_avx512(a, b, out, chunks),
        BinaryOp::Pow => {
            // Pow has no direct SIMD instruction, use scalar
            binary_scalar_f32(op, a, b, out, len);
            return;
        }
    }

    // Handle tail with scalar
    if remainder > 0 {
        let offset = chunks * F32_LANES;
        binary_scalar_f32(op, a.add(offset), b.add(offset), out.add(offset), remainder);
    }
}

/// AVX-512 binary operation for f64
///
/// # Safety
/// - CPU must support AVX-512F
/// - All pointers must be valid for `len` elements
#[target_feature(enable = "avx512f")]
pub unsafe fn binary_f64(op: BinaryOp, a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    match op {
        BinaryOp::Add => binary_add_f64_avx512(a, b, out, chunks),
        BinaryOp::Sub => binary_sub_f64_avx512(a, b, out, chunks),
        BinaryOp::Mul => binary_mul_f64_avx512(a, b, out, chunks),
        BinaryOp::Div => binary_div_f64_avx512(a, b, out, chunks),
        BinaryOp::Max => binary_max_f64_avx512(a, b, out, chunks),
        BinaryOp::Min => binary_min_f64_avx512(a, b, out, chunks),
        BinaryOp::Pow => {
            binary_scalar_f64(op, a, b, out, len);
            return;
        }
    }

    // Handle tail with scalar
    if remainder > 0 {
        let offset = chunks * F64_LANES;
        binary_scalar_f64(op, a.add(offset), b.add(offset), out.add(offset), remainder);
    }
}

// ============================================================================
// f32 kernels
// ============================================================================

#[target_feature(enable = "avx512f")]
unsafe fn binary_add_f32_avx512(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));
        let vr = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn binary_sub_f32_avx512(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));
        let vr = _mm512_sub_ps(va, vb);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn binary_mul_f32_avx512(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));
        let vr = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn binary_div_f32_avx512(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));
        let vr = _mm512_div_ps(va, vb);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn binary_max_f32_avx512(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));
        let vr = _mm512_max_ps(va, vb);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn binary_min_f32_avx512(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));
        let vr = _mm512_min_ps(va, vb);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

// ============================================================================
// f64 kernels
// ============================================================================

#[target_feature(enable = "avx512f")]
unsafe fn binary_add_f64_avx512(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vb = _mm512_loadu_pd(b.add(offset));
        let vr = _mm512_add_pd(va, vb);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn binary_sub_f64_avx512(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vb = _mm512_loadu_pd(b.add(offset));
        let vr = _mm512_sub_pd(va, vb);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn binary_mul_f64_avx512(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vb = _mm512_loadu_pd(b.add(offset));
        let vr = _mm512_mul_pd(va, vb);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn binary_div_f64_avx512(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vb = _mm512_loadu_pd(b.add(offset));
        let vr = _mm512_div_pd(va, vb);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn binary_max_f64_avx512(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vb = _mm512_loadu_pd(b.add(offset));
        let vr = _mm512_max_pd(va, vb);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn binary_min_f64_avx512(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vb = _mm512_loadu_pd(b.add(offset));
        let vr = _mm512_min_pd(va, vb);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}
