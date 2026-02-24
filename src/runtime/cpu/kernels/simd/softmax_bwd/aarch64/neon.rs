//! NEON softmax backward kernels for ARM64.
//!
//! Fused 2-pass: SIMD dot product, then SIMD elementwise output * (grad - dot).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::super::math::aarch64::neon::{hsum_f32, hsum_f64};

const F32_LANES: usize = 4;
const F64_LANES: usize = 2;

/// NEON softmax backward for f32.
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - All pointers must point to `outer_size * dim_size` valid f32 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn softmax_bwd_f32(
    grad: *const f32,
    output: *const f32,
    d_input: *mut f32,
    outer_size: usize,
    dim_size: usize,
) {
    let chunks = dim_size / F32_LANES;
    let remainder = dim_size % F32_LANES;

    for o in 0..outer_size {
        let g_base = grad.add(o * dim_size);
        let o_base = output.add(o * dim_size);
        let d_base = d_input.add(o * dim_size);

        // Pass 1: SIMD dot product
        let mut dot_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let g = vld1q_f32(g_base.add(offset));
            let out = vld1q_f32(o_base.add(offset));
            dot_acc = vfmaq_f32(dot_acc, g, out);
        }
        let mut dot = hsum_f32(dot_acc);

        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            dot += *g_base.add(offset) * *o_base.add(offset);
        }

        // Pass 2: d_input = output * (grad - dot)
        let v_dot = vdupq_n_f32(dot);
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let g = vld1q_f32(g_base.add(offset));
            let out = vld1q_f32(o_base.add(offset));
            let shifted = vsubq_f32(g, v_dot);
            let result = vmulq_f32(out, shifted);
            vst1q_f32(d_base.add(offset), result);
        }

        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            *d_base.add(offset) = *o_base.add(offset) * (*g_base.add(offset) - dot);
        }
    }
}

/// NEON softmax backward for f64.
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - All pointers must point to `outer_size * dim_size` valid f64 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn softmax_bwd_f64(
    grad: *const f64,
    output: *const f64,
    d_input: *mut f64,
    outer_size: usize,
    dim_size: usize,
) {
    let chunks = dim_size / F64_LANES;
    let remainder = dim_size % F64_LANES;

    for o in 0..outer_size {
        let g_base = grad.add(o * dim_size);
        let o_base = output.add(o * dim_size);
        let d_base = d_input.add(o * dim_size);

        // Pass 1: SIMD dot product
        let mut dot_acc = vdupq_n_f64(0.0);
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let g = vld1q_f64(g_base.add(offset));
            let out = vld1q_f64(o_base.add(offset));
            dot_acc = vfmaq_f64(dot_acc, g, out);
        }
        let mut dot = hsum_f64(dot_acc);

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            dot += *g_base.add(offset) * *o_base.add(offset);
        }

        // Pass 2: d_input = output * (grad - dot)
        let v_dot = vdupq_n_f64(dot);
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let g = vld1q_f64(g_base.add(offset));
            let out = vld1q_f64(o_base.add(offset));
            let shifted = vsubq_f64(g, v_dot);
            let result = vmulq_f64(out, shifted);
            vst1q_f64(d_base.add(offset), result);
        }

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            *d_base.add(offset) = *o_base.add(offset) * (*g_base.add(offset) - dot);
        }
    }
}
