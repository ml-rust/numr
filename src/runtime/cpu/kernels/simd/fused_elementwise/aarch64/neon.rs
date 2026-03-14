//! NEON fused elementwise kernels (128-bit)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::{
    fused_add_mul_scalar_f32 as fused_add_mul_fallback_f32,
    fused_add_mul_scalar_f64 as fused_add_mul_fallback_f64,
    fused_mul_add_scalar_f32 as fused_mul_add_fallback_f32,
    fused_mul_add_scalar_f64 as fused_mul_add_fallback_f64, fused_mul_add_scalar_loop_f32,
    fused_mul_add_scalar_loop_f64,
};

const F32_LANES: usize = 4;
const F64_LANES: usize = 2;

/// NEON fused_mul_add for f32: out = a * b + c
#[target_feature(enable = "neon")]
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
        let va = vld1q_f32(a.add(offset));
        let vb = vld1q_f32(b.add(offset));
        let vc = vld1q_f32(c.add(offset));
        // vfmaq_f32: vc + va * vb
        let result = vfmaq_f32(vc, va, vb);
        vst1q_f32(out.add(offset), result);
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

/// NEON fused_mul_add for f64: out = a * b + c
#[target_feature(enable = "neon")]
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
        let va = vld1q_f64(a.add(offset));
        let vb = vld1q_f64(b.add(offset));
        let vc = vld1q_f64(c.add(offset));
        let result = vfmaq_f64(vc, va, vb);
        vst1q_f64(out.add(offset), result);
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

/// NEON fused_add_mul for f32: out = (a + b) * c
#[target_feature(enable = "neon")]
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
        let va = vld1q_f32(a.add(offset));
        let vb = vld1q_f32(b.add(offset));
        let vc = vld1q_f32(c.add(offset));
        let sum = vaddq_f32(va, vb);
        let result = vmulq_f32(sum, vc);
        vst1q_f32(out.add(offset), result);
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

/// NEON fused_add_mul for f64: out = (a + b) * c
#[target_feature(enable = "neon")]
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
        let va = vld1q_f64(a.add(offset));
        let vb = vld1q_f64(b.add(offset));
        let vc = vld1q_f64(c.add(offset));
        let sum = vaddq_f64(va, vb);
        let result = vmulq_f64(sum, vc);
        vst1q_f64(out.add(offset), result);
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

/// NEON fused_mul_add_scalar for f32: out = a * scale + bias
#[target_feature(enable = "neon")]
pub unsafe fn fused_mul_add_scalar_f32(
    a: *const f32,
    scale: f32,
    bias: f32,
    out: *mut f32,
    len: usize,
) {
    let chunks = len / F32_LANES;
    let vscale = vdupq_n_f32(scale);
    let vbias = vdupq_n_f32(bias);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let result = vfmaq_f32(vbias, va, vscale);
        vst1q_f32(out.add(offset), result);
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

/// NEON fused_mul_add_scalar for f64: out = a * scale + bias
#[target_feature(enable = "neon")]
pub unsafe fn fused_mul_add_scalar_f64(
    a: *const f64,
    scale: f64,
    bias: f64,
    out: *mut f64,
    len: usize,
) {
    let chunks = len / F64_LANES;
    let vscale = vdupq_n_f64(scale);
    let vbias = vdupq_n_f64(bias);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let result = vfmaq_f64(vbias, va, vscale);
        vst1q_f64(out.add(offset), result);
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
