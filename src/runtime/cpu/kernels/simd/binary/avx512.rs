//! AVX-512 binary operation kernels
//!
//! Processes 16 f32s or 8 f64s per iteration using 512-bit vectors.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::math::avx512::{
    exp_f32 as exp_vec_f32, exp_f64 as exp_vec_f64, log_f32 as log_vec_f32, log_f64 as log_vec_f64,
};
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
        BinaryOp::Pow => binary_pow_f32_avx512(a, b, out, chunks),
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
        BinaryOp::Pow => binary_pow_f64_avx512(a, b, out, chunks),
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

// ============================================================================
// Pow kernels using exp(b * log(a))
// ============================================================================

/// SIMD pow for f32: a^b = exp(b * log(a))
///
/// # Algorithm
/// Uses the identity: a^b = exp(b * log(a))
///
/// # Edge Cases (Documented Limitations)
/// This implementation prioritizes performance over edge case handling:
/// - **Negative base (a < 0)**: Returns NaN (log of negative is undefined)
/// - **Zero base (a = 0)**: Returns NaN or -inf depending on b (log(0) = -inf)
/// - **Overflow**: Large b*log(a) may overflow exp()
///
/// For applications requiring IEEE 754 compliant pow() behavior, use scalar
/// fallback or handle edge cases before calling this function.
///
/// # Accuracy
/// Relative error < 1e-3 due to compound error from exp(b*log(a))
#[target_feature(enable = "avx512f")]
unsafe fn binary_pow_f32_avx512(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));
        // pow(a, b) = exp(b * log(a))
        let log_a = log_vec_f32(va);
        let b_log_a = _mm512_mul_ps(vb, log_a);
        let vr = exp_vec_f32(b_log_a);
        _mm512_storeu_ps(out.add(offset), vr);
    }
}

/// SIMD pow for f64: a^b = exp(b * log(a))
///
/// See `binary_pow_f32_avx512` for algorithm and edge case documentation.
///
/// # Accuracy
/// Relative error < 1e-4 due to compound error from exp(b*log(a))
#[target_feature(enable = "avx512f")]
unsafe fn binary_pow_f64_avx512(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vb = _mm512_loadu_pd(b.add(offset));
        // pow(a, b) = exp(b * log(a))
        let log_a = log_vec_f64(va);
        let b_log_a = _mm512_mul_pd(vb, log_a);
        let vr = exp_vec_f64(b_log_a);
        _mm512_storeu_pd(out.add(offset), vr);
    }
}
