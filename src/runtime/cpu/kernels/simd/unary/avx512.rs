//! AVX-512 unary operation kernels
//!
//! Processes 16 f32s or 8 f64s per iteration using 512-bit vectors.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{relu_scalar_f32, relu_scalar_f64, unary_scalar_f32, unary_scalar_f64};
use crate::ops::UnaryOp;

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

/// AVX-512 unary operation for f32
#[target_feature(enable = "avx512f")]
pub unsafe fn unary_f32(op: UnaryOp, a: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    match op {
        UnaryOp::Neg => unary_neg_f32(a, out, chunks),
        UnaryOp::Abs => unary_abs_f32(a, out, chunks),
        UnaryOp::Sqrt => unary_sqrt_f32(a, out, chunks),
        UnaryOp::Square => unary_square_f32(a, out, chunks),
        UnaryOp::Recip => unary_recip_f32(a, out, chunks),
        UnaryOp::Floor => unary_floor_f32(a, out, chunks),
        UnaryOp::Ceil => unary_ceil_f32(a, out, chunks),
        UnaryOp::Round => unary_round_f32(a, out, chunks),
        _ => {
            // Unsupported ops use full scalar path
            unary_scalar_f32(op, a, out, len);
            return;
        }
    }

    // Handle tail
    if remainder > 0 {
        let offset = chunks * F32_LANES;
        unary_scalar_f32(op, a.add(offset), out.add(offset), remainder);
    }
}

/// AVX-512 unary operation for f64
#[target_feature(enable = "avx512f")]
pub unsafe fn unary_f64(op: UnaryOp, a: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    match op {
        UnaryOp::Neg => unary_neg_f64(a, out, chunks),
        UnaryOp::Abs => unary_abs_f64(a, out, chunks),
        UnaryOp::Sqrt => unary_sqrt_f64(a, out, chunks),
        UnaryOp::Square => unary_square_f64(a, out, chunks),
        UnaryOp::Recip => unary_recip_f64(a, out, chunks),
        UnaryOp::Floor => unary_floor_f64(a, out, chunks),
        UnaryOp::Ceil => unary_ceil_f64(a, out, chunks),
        UnaryOp::Round => unary_round_f64(a, out, chunks),
        _ => {
            unary_scalar_f64(op, a, out, len);
            return;
        }
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        unary_scalar_f64(op, a.add(offset), out.add(offset), remainder);
    }
}

/// AVX-512 ReLU for f32
#[target_feature(enable = "avx512f")]
pub unsafe fn relu_f32(a: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;
    let zero = _mm512_setzero_ps();

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vr = _mm512_max_ps(va, zero);
        _mm512_storeu_ps(out.add(offset), vr);
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        relu_scalar_f32(a.add(offset), out.add(offset), remainder);
    }
}

/// AVX-512 ReLU for f64
#[target_feature(enable = "avx512f")]
pub unsafe fn relu_f64(a: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;
    let zero = _mm512_setzero_pd();

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vr = _mm512_max_pd(va, zero);
        _mm512_storeu_pd(out.add(offset), vr);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        relu_scalar_f64(a.add(offset), out.add(offset), remainder);
    }
}

// ============================================================================
// f32 kernels
// ============================================================================

#[target_feature(enable = "avx512f")]
unsafe fn unary_neg_f32(a: *const f32, out: *mut f32, chunks: usize) {
    let zero = _mm512_setzero_ps();
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vr = _mm512_sub_ps(zero, va);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_abs_f32(a: *const f32, out: *mut f32, chunks: usize) {
    let mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF)); // Clear sign bit
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vr = _mm512_and_ps(va, mask);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_sqrt_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vr = _mm512_sqrt_ps(va);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_square_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vr = _mm512_mul_ps(va, va);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_recip_f32(a: *const f32, out: *mut f32, chunks: usize) {
    let one = _mm512_set1_ps(1.0);
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vr = _mm512_div_ps(one, va);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_floor_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        // _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC = 0x09
        let vr = _mm512_roundscale_ps::<0x09>(va);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_ceil_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        // _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC = 0x0A
        let vr = _mm512_roundscale_ps::<0x0A>(va);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_round_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        // _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC = 0x08
        let vr = _mm512_roundscale_ps::<0x08>(va);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

// ============================================================================
// f64 kernels
// ============================================================================

#[target_feature(enable = "avx512f")]
unsafe fn unary_neg_f64(a: *const f64, out: *mut f64, chunks: usize) {
    let zero = _mm512_setzero_pd();
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vr = _mm512_sub_pd(zero, va);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_abs_f64(a: *const f64, out: *mut f64, chunks: usize) {
    let mask = _mm512_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vr = _mm512_and_pd(va, mask);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_sqrt_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vr = _mm512_sqrt_pd(va);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_square_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vr = _mm512_mul_pd(va, va);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_recip_f64(a: *const f64, out: *mut f64, chunks: usize) {
    let one = _mm512_set1_pd(1.0);
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vr = _mm512_div_pd(one, va);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_floor_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        // _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC = 0x09
        let vr = _mm512_roundscale_pd::<0x09>(va);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_ceil_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        // _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC = 0x0A
        let vr = _mm512_roundscale_pd::<0x0A>(va);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_round_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vr = _mm512_roundscale_pd::<0x08>(va);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}
