//! AVX-512 clamp kernels
//!
//! Uses SIMD min/max operations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{clamp_scalar_f32, clamp_scalar_f64};

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

/// AVX-512 clamp for f32
#[target_feature(enable = "avx512f")]
pub unsafe fn clamp_f32(a: *const f32, out: *mut f32, len: usize, min_val: f32, max_val: f32) {
    let chunks = len / F32_LANES;
    let v_min = _mm512_set1_ps(min_val);
    let v_max = _mm512_set1_ps(max_val);

    for c in 0..chunks {
        let offset = c * F32_LANES;
        let v = _mm512_loadu_ps(a.add(offset));

        // clamp = min(max(v, min_val), max_val)
        let clamped = _mm512_min_ps(_mm512_max_ps(v, v_min), v_max);

        _mm512_storeu_ps(out.add(offset), clamped);
    }

    // Scalar tail
    let processed = chunks * F32_LANES;
    if processed < len {
        clamp_scalar_f32(
            a.add(processed),
            out.add(processed),
            len - processed,
            min_val,
            max_val,
        );
    }
}

/// AVX-512 clamp for f64
#[target_feature(enable = "avx512f")]
pub unsafe fn clamp_f64(a: *const f64, out: *mut f64, len: usize, min_val: f64, max_val: f64) {
    let chunks = len / F64_LANES;
    let v_min = _mm512_set1_pd(min_val);
    let v_max = _mm512_set1_pd(max_val);

    for c in 0..chunks {
        let offset = c * F64_LANES;
        let v = _mm512_loadu_pd(a.add(offset));

        let clamped = _mm512_min_pd(_mm512_max_pd(v, v_min), v_max);

        _mm512_storeu_pd(out.add(offset), clamped);
    }

    // Scalar tail
    let processed = chunks * F64_LANES;
    if processed < len {
        clamp_scalar_f64(
            a.add(processed),
            out.add(processed),
            len - processed,
            min_val,
            max_val,
        );
    }
}

// Suppress unused warnings
const _: () = {
    let _ = clamp_scalar_f32 as unsafe fn(*const f32, *mut f32, usize, f32, f32);
    let _ = clamp_scalar_f64 as unsafe fn(*const f64, *mut f64, usize, f64, f64);
};
