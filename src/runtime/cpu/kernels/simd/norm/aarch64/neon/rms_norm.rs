//! NEON RMS normalization kernels

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::super::super::math::aarch64::neon::{hsum_f32, hsum_f64};
use super::{F32_LANES, F64_LANES};

/// NEON RMS normalization for f32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `input` and `out` must point to `batch_size * hidden_size` valid f32 elements
/// - `weight` must point to `hidden_size` valid f32 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn rms_norm_f32(
    input: *const f32,
    weight: *const f32,
    out: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let chunks = hidden_size / F32_LANES;
    let remainder = hidden_size % F32_LANES;

    for b in 0..batch_size {
        let base = input.add(b * hidden_size);
        let out_base = out.add(b * hidden_size);

        // Phase 1: Sum of squares using FMA
        let mut ss_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(base.add(i * F32_LANES));
            ss_acc = vfmaq_f32(ss_acc, v, v);
        }
        let mut sum_sq = hsum_f32(ss_acc) as f64;

        // Scalar tail for sum of squares
        for i in 0..remainder {
            let v = *base.add(chunks * F32_LANES + i) as f64;
            sum_sq += v * v;
        }

        // Compute inverse RMS in f64 for precision (matches llama.cpp)
        let inv_rms = (1.0f64 / (sum_sq / hidden_size as f64 + eps as f64).sqrt()) as f32;
        let v_inv_rms = vdupq_n_f32(inv_rms);

        // Phase 2: Apply normalization and weight
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let v_in = vld1q_f32(base.add(offset));
            let v_w = vld1q_f32(weight.add(offset));
            let result = vmulq_f32(vmulq_f32(v_in, v_inv_rms), v_w);
            vst1q_f32(out_base.add(offset), result);
        }

        // Scalar tail for normalization
        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            *out_base.add(offset) = *base.add(offset) * inv_rms * *weight.add(offset);
        }
    }
}

/// NEON RMS normalization for f64
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `input` and `out` must point to `batch_size * hidden_size` valid f64 elements
/// - `weight` must point to `hidden_size` valid f64 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn rms_norm_f64(
    input: *const f64,
    weight: *const f64,
    out: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    let chunks = hidden_size / F64_LANES;
    let remainder = hidden_size % F64_LANES;

    for b in 0..batch_size {
        let base = input.add(b * hidden_size);
        let out_base = out.add(b * hidden_size);

        // Phase 1: Sum of squares
        let mut ss_acc = vdupq_n_f64(0.0);
        for i in 0..chunks {
            let v = vld1q_f64(base.add(i * F64_LANES));
            ss_acc = vfmaq_f64(ss_acc, v, v);
        }
        let mut sum_sq = hsum_f64(ss_acc);

        for i in 0..remainder {
            let v = *base.add(chunks * F64_LANES + i);
            sum_sq += v * v;
        }

        let inv_rms = 1.0 / (sum_sq / hidden_size as f64 + eps).sqrt();
        let v_inv_rms = vdupq_n_f64(inv_rms);

        // Phase 2: Apply normalization and weight
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let v_in = vld1q_f64(base.add(offset));
            let v_w = vld1q_f64(weight.add(offset));
            let result = vmulq_f64(vmulq_f64(v_in, v_inv_rms), v_w);
            vst1q_f64(out_base.add(offset), result);
        }

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            *out_base.add(offset) = *base.add(offset) * inv_rms * *weight.add(offset);
        }
    }
}
