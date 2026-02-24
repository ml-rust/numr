//! AVX-512 fused elementwise kernels (512-bit)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::{
    fused_add_mul_scalar_f32 as fused_add_mul_fallback_f32,
    fused_add_mul_scalar_f64 as fused_add_mul_fallback_f64,
    fused_mul_add_scalar_f32 as fused_mul_add_fallback_f32,
    fused_mul_add_scalar_f64 as fused_mul_add_fallback_f64, fused_mul_add_scalar_loop_f32,
    fused_mul_add_scalar_loop_f64,
};

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

/// AVX-512 fused_mul_add for f32: out = a * b + c
#[target_feature(enable = "avx512f")]
pub unsafe fn fused_mul_add_f32(
    a: *const f32,
    b: *const f32,
    c: *const f32,
    out: *mut f32,
    len: usize,
) {
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));
        let vc = _mm512_loadu_ps(c.add(offset));
        let result = _mm512_fmadd_ps(va, vb, vc);
        _mm512_storeu_ps(out.add(offset), result);
    }

    let processed = chunks * F32_LANES;
    if processed < len {
        fused_mul_add_fallback_f32(
            a.add(processed),
            b.add(processed),
            c.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

/// AVX-512 fused_mul_add for f64: out = a * b + c
#[target_feature(enable = "avx512f")]
pub unsafe fn fused_mul_add_f64(
    a: *const f64,
    b: *const f64,
    c: *const f64,
    out: *mut f64,
    len: usize,
) {
    let chunks = len / F64_LANES;

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vb = _mm512_loadu_pd(b.add(offset));
        let vc = _mm512_loadu_pd(c.add(offset));
        let result = _mm512_fmadd_pd(va, vb, vc);
        _mm512_storeu_pd(out.add(offset), result);
    }

    let processed = chunks * F64_LANES;
    if processed < len {
        fused_mul_add_fallback_f64(
            a.add(processed),
            b.add(processed),
            c.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

/// AVX-512 fused_add_mul for f32: out = (a + b) * c
#[target_feature(enable = "avx512f")]
pub unsafe fn fused_add_mul_f32(
    a: *const f32,
    b: *const f32,
    c: *const f32,
    out: *mut f32,
    len: usize,
) {
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let vb = _mm512_loadu_ps(b.add(offset));
        let vc = _mm512_loadu_ps(c.add(offset));
        let sum = _mm512_add_ps(va, vb);
        let result = _mm512_mul_ps(sum, vc);
        _mm512_storeu_ps(out.add(offset), result);
    }

    let processed = chunks * F32_LANES;
    if processed < len {
        fused_add_mul_fallback_f32(
            a.add(processed),
            b.add(processed),
            c.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

/// AVX-512 fused_add_mul for f64: out = (a + b) * c
#[target_feature(enable = "avx512f")]
pub unsafe fn fused_add_mul_f64(
    a: *const f64,
    b: *const f64,
    c: *const f64,
    out: *mut f64,
    len: usize,
) {
    let chunks = len / F64_LANES;

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let vb = _mm512_loadu_pd(b.add(offset));
        let vc = _mm512_loadu_pd(c.add(offset));
        let sum = _mm512_add_pd(va, vb);
        let result = _mm512_mul_pd(sum, vc);
        _mm512_storeu_pd(out.add(offset), result);
    }

    let processed = chunks * F64_LANES;
    if processed < len {
        fused_add_mul_fallback_f64(
            a.add(processed),
            b.add(processed),
            c.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

/// AVX-512 fused_mul_add_scalar for f32: out = a * scale + bias
#[target_feature(enable = "avx512f")]
pub unsafe fn fused_mul_add_scalar_f32(
    a: *const f32,
    scale: f32,
    bias: f32,
    out: *mut f32,
    len: usize,
) {
    let chunks = len / F32_LANES;
    let vscale = _mm512_set1_ps(scale);
    let vbias = _mm512_set1_ps(bias);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm512_loadu_ps(a.add(offset));
        let result = _mm512_fmadd_ps(va, vscale, vbias);
        _mm512_storeu_ps(out.add(offset), result);
    }

    let processed = chunks * F32_LANES;
    if processed < len {
        fused_mul_add_scalar_loop_f32(
            a.add(processed),
            scale,
            bias,
            out.add(processed),
            len - processed,
        );
    }
}

/// AVX-512 fused_mul_add_scalar for f64: out = a * scale + bias
#[target_feature(enable = "avx512f")]
pub unsafe fn fused_mul_add_scalar_f64(
    a: *const f64,
    scale: f64,
    bias: f64,
    out: *mut f64,
    len: usize,
) {
    let chunks = len / F64_LANES;
    let vscale = _mm512_set1_pd(scale);
    let vbias = _mm512_set1_pd(bias);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm512_loadu_pd(a.add(offset));
        let result = _mm512_fmadd_pd(va, vscale, vbias);
        _mm512_storeu_pd(out.add(offset), result);
    }

    let processed = chunks * F64_LANES;
    if processed < len {
        fused_mul_add_scalar_loop_f64(
            a.add(processed),
            scale,
            bias,
            out.add(processed),
            len - processed,
        );
    }
}
