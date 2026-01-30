//! AVX-512 unary operation kernels
//!
//! Processes 16 f32s or 8 f64s per iteration using 512-bit vectors.
//!
//! # Streaming Stores
//!
//! For large arrays (> 1MB), this module provides streaming variants that use
//! non-temporal stores (`_mm512_stream_ps`) to bypass the cache.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::math::avx512::{
    cos_f32 as cos_vec_f32, cos_f64 as cos_vec_f64, exp_f32 as exp_vec_f32, exp_f64 as exp_vec_f64,
    log_f32 as log_vec_f32, log_f64 as log_vec_f64, sin_f32 as sin_vec_f32, sin_f64 as sin_vec_f64,
    tan_f32 as tan_vec_f32, tan_f64 as tan_vec_f64, tanh_f32 as tanh_vec_f32,
    tanh_f64 as tanh_vec_f64,
};
use super::super::streaming::{is_aligned_avx512, should_stream_f32, should_stream_f64};
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
        UnaryOp::Exp => unary_exp_f32(a, out, chunks),
        UnaryOp::Log => unary_log_f32(a, out, chunks),
        UnaryOp::Sin => unary_sin_f32(a, out, chunks),
        UnaryOp::Cos => unary_cos_f32(a, out, chunks),
        UnaryOp::Tan => unary_tan_f32(a, out, chunks),
        UnaryOp::Tanh => unary_tanh_f32(a, out, chunks),
        UnaryOp::Square => unary_square_f32(a, out, chunks),
        UnaryOp::Recip => unary_recip_f32(a, out, chunks),
        UnaryOp::Floor => unary_floor_f32(a, out, chunks),
        UnaryOp::Ceil => unary_ceil_f32(a, out, chunks),
        UnaryOp::Round => unary_round_f32(a, out, chunks),
        UnaryOp::Sign => unary_sign_f32(a, out, chunks),
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
        UnaryOp::Exp => unary_exp_f64(a, out, chunks),
        UnaryOp::Log => unary_log_f64(a, out, chunks),
        UnaryOp::Sin => unary_sin_f64(a, out, chunks),
        UnaryOp::Cos => unary_cos_f64(a, out, chunks),
        UnaryOp::Tan => unary_tan_f64(a, out, chunks),
        UnaryOp::Tanh => unary_tanh_f64(a, out, chunks),
        UnaryOp::Square => unary_square_f64(a, out, chunks),
        UnaryOp::Recip => unary_recip_f64(a, out, chunks),
        UnaryOp::Floor => unary_floor_f64(a, out, chunks),
        UnaryOp::Ceil => unary_ceil_f64(a, out, chunks),
        UnaryOp::Round => unary_round_f64(a, out, chunks),
        UnaryOp::Sign => unary_sign_f64(a, out, chunks),
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        unary_scalar_f64(op, a.add(offset), out.add(offset), remainder);
    }
}

/// AVX-512 ReLU for f32
///
/// Uses streaming (non-temporal) stores for arrays > 1MB when output is aligned.
#[target_feature(enable = "avx512f")]
pub unsafe fn relu_f32(a: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;
    let zero = _mm512_setzero_ps();

    // Use streaming stores for large aligned arrays
    let use_streaming = should_stream_f32(len) && is_aligned_avx512(out);

    if use_streaming {
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let va = _mm512_loadu_ps(a.add(offset));
            let vr = _mm512_max_ps(va, zero);
            _mm512_stream_ps(out.add(offset), vr);
        }
        _mm_sfence();
    } else {
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let va = _mm512_loadu_ps(a.add(offset));
            let vr = _mm512_max_ps(va, zero);
            _mm512_storeu_ps(out.add(offset), vr);
        }
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        relu_scalar_f32(a.add(offset), out.add(offset), remainder);
    }
}

/// AVX-512 ReLU for f64
///
/// Uses streaming (non-temporal) stores for arrays > 1MB when output is aligned.
#[target_feature(enable = "avx512f")]
pub unsafe fn relu_f64(a: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;
    let zero = _mm512_setzero_pd();

    // Use streaming stores for large aligned arrays
    let use_streaming = should_stream_f64(len) && is_aligned_avx512(out);

    if use_streaming {
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let va = _mm512_loadu_pd(a.add(offset));
            let vr = _mm512_max_pd(va, zero);
            _mm512_stream_pd(out.add(offset), vr);
        }
        _mm_sfence();
    } else {
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let va = _mm512_loadu_pd(a.add(offset));
            let vr = _mm512_max_pd(va, zero);
            _mm512_storeu_pd(out.add(offset), vr);
        }
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

#[target_feature(enable = "avx512f")]
unsafe fn unary_exp_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vr = exp_vec_f32(va);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_tanh_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vr = tanh_vec_f32(va);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_log_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vr = log_vec_f32(va);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_sin_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vr = sin_vec_f32(va);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_cos_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vr = cos_vec_f32(va);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_tan_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vr = tan_vec_f32(va);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_sign_f32(a: *const f32, out: *mut f32, chunks: usize) {
    let zero = _mm512_setzero_ps();
    let one = _mm512_set1_ps(1.0);
    let neg_one = _mm512_set1_ps(-1.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        // sign(x) = (x > 0) ? 1 : ((x < 0) ? -1 : 0)
        let pos_mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(va, zero);
        let neg_mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(va, zero);
        let pos_part = _mm512_maskz_mov_ps(pos_mask, one);
        let neg_part = _mm512_maskz_mov_ps(neg_mask, neg_one);
        let vr = _mm512_add_ps(pos_part, neg_part);
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

#[target_feature(enable = "avx512f")]
unsafe fn unary_exp_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vr = exp_vec_f64(va);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_tanh_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vr = tanh_vec_f64(va);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_log_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vr = log_vec_f64(va);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_sin_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vr = sin_vec_f64(va);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_cos_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vr = cos_vec_f64(va);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_tan_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vr = tan_vec_f64(va);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn unary_sign_f64(a: *const f64, out: *mut f64, chunks: usize) {
    let zero = _mm512_setzero_pd();
    let one = _mm512_set1_pd(1.0);
    let neg_one = _mm512_set1_pd(-1.0);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        // sign(x) = (x > 0) ? 1 : ((x < 0) ? -1 : 0)
        let pos_mask = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(va, zero);
        let neg_mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(va, zero);
        let pos_part = _mm512_maskz_mov_pd(pos_mask, one);
        let neg_part = _mm512_maskz_mov_pd(neg_mask, neg_one);
        let vr = _mm512_add_pd(pos_part, neg_part);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}
