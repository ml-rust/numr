//! NEON fused add + RMS normalization kernels (forward and backward)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::super::super::math::aarch64::neon::{hsum_f32, hsum_f64};
use super::{F32_LANES, F64_LANES};

/// NEON Fused Add + RMS Normalization for f32
///
/// Computes: output = (input + residual) * rsqrt(mean((input + residual)^2) + eps) * weight
/// Stores intermediate (input + residual) in pre_norm for backward pass.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_rms_norm_f32(
    input: *const f32,
    residual: *const f32,
    weight: *const f32,
    out: *mut f32,
    pre_norm: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let chunks = hidden_size / F32_LANES;
    let remainder = hidden_size % F32_LANES;

    for b in 0..batch_size {
        let base = input.add(b * hidden_size);
        let res_base = residual.add(b * hidden_size);
        let pn_base = pre_norm.add(b * hidden_size);
        let out_base = out.add(b * hidden_size);

        // Phase 1: Add input + residual, store in pre_norm, accumulate sum of squares
        let mut ss_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let v_in = vld1q_f32(base.add(offset));
            let v_res = vld1q_f32(res_base.add(offset));
            let pn = vaddq_f32(v_in, v_res);
            vst1q_f32(pn_base.add(offset), pn);
            ss_acc = vfmaq_f32(ss_acc, pn, pn);
        }
        let mut sum_sq = hsum_f32(ss_acc);

        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            let pn = *base.add(offset) + *res_base.add(offset);
            *pn_base.add(offset) = pn;
            sum_sq += pn * pn;
        }

        let inv_rms = 1.0 / (sum_sq / hidden_size as f32 + eps).sqrt();
        let v_inv_rms = vdupq_n_f32(inv_rms);

        // Phase 2: Apply normalization and weight
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let pn = vld1q_f32(pn_base.add(offset));
            let v_w = vld1q_f32(weight.add(offset));
            let result = vmulq_f32(vmulq_f32(pn, v_inv_rms), v_w);
            vst1q_f32(out_base.add(offset), result);
        }

        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            let pn = *pn_base.add(offset);
            let w = *weight.add(offset);
            *out_base.add(offset) = pn * inv_rms * w;
        }
    }
}

/// NEON Fused Add + RMS Normalization for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_rms_norm_f64(
    input: *const f64,
    residual: *const f64,
    weight: *const f64,
    out: *mut f64,
    pre_norm: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    let chunks = hidden_size / F64_LANES;
    let remainder = hidden_size % F64_LANES;

    for b in 0..batch_size {
        let base = input.add(b * hidden_size);
        let res_base = residual.add(b * hidden_size);
        let pn_base = pre_norm.add(b * hidden_size);
        let out_base = out.add(b * hidden_size);

        let mut ss_acc = vdupq_n_f64(0.0);
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let v_in = vld1q_f64(base.add(offset));
            let v_res = vld1q_f64(res_base.add(offset));
            let pn = vaddq_f64(v_in, v_res);
            vst1q_f64(pn_base.add(offset), pn);
            ss_acc = vfmaq_f64(ss_acc, pn, pn);
        }
        let mut sum_sq = hsum_f64(ss_acc);

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            let pn = *base.add(offset) + *res_base.add(offset);
            *pn_base.add(offset) = pn;
            sum_sq += pn * pn;
        }

        let inv_rms = 1.0 / (sum_sq / hidden_size as f64 + eps).sqrt();
        let v_inv_rms = vdupq_n_f64(inv_rms);

        for i in 0..chunks {
            let offset = i * F64_LANES;
            let pn = vld1q_f64(pn_base.add(offset));
            let v_w = vld1q_f64(weight.add(offset));
            let result = vmulq_f64(vmulq_f64(pn, v_inv_rms), v_w);
            vst1q_f64(out_base.add(offset), result);
        }

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            let pn = *pn_base.add(offset);
            let w = *weight.add(offset);
            *out_base.add(offset) = pn * inv_rms * w;
        }
    }
}

/// NEON Fused Add + RMS Norm Backward for f32
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_rms_norm_bwd_f32(
    grad: *const f32,
    pre_norm: *const f32,
    weight: *const f32,
    d_input_residual: *mut f32,
    d_weight: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let chunks = hidden_size / F32_LANES;
    let remainder = hidden_size % F32_LANES;

    for b in 0..batch_size {
        let pn_base = pre_norm.add(b * hidden_size);
        let grad_base = grad.add(b * hidden_size);
        let d_ir_base = d_input_residual.add(b * hidden_size);

        // Recompute mean square from pre_norm
        let mut acc_sq = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let pn = vld1q_f32(pn_base.add(offset));
            acc_sq = vfmaq_f32(acc_sq, pn, pn);
        }
        let mut sum_sq = hsum_f32(acc_sq);

        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            let pn = *pn_base.add(offset);
            sum_sq += pn * pn;
        }

        let mean_sq = sum_sq / hidden_size as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        // Compute dot = sum(grad * weight * pre_norm)
        let mut dot_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let g = vld1q_f32(grad_base.add(offset));
            let w = vld1q_f32(weight.add(offset));
            let pn = vld1q_f32(pn_base.add(offset));
            let gw = vmulq_f32(g, w);
            dot_acc = vfmaq_f32(dot_acc, gw, pn);
        }
        let mut dot = hsum_f32(dot_acc);

        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            let g = *grad_base.add(offset);
            let w = *weight.add(offset);
            let pn = *pn_base.add(offset);
            dot += g * w * pn;
        }

        let coeff = dot * inv_rms / (hidden_size as f32 * (mean_sq + eps));
        let v_inv_rms = vdupq_n_f32(inv_rms);
        let v_coeff = vdupq_n_f32(coeff);

        // Compute d_input_residual and accumulate d_weight
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let g = vld1q_f32(grad_base.add(offset));
            let w = vld1q_f32(weight.add(offset));
            let pn = vld1q_f32(pn_base.add(offset));

            // d_ir = (g*w - pn*coeff) * inv_rms
            let gw = vmulq_f32(g, w);
            let pn_coeff = vmulq_f32(pn, v_coeff);
            let diff = vsubq_f32(gw, pn_coeff);
            let d_ir = vmulq_f32(diff, v_inv_rms);
            vst1q_f32(d_ir_base.add(offset), d_ir);

            // d_weight += g * pn * inv_rms
            let dw_old = vld1q_f32(d_weight.add(offset));
            let gp = vmulq_f32(g, pn);
            let gp_inv = vmulq_f32(gp, v_inv_rms);
            let dw_new = vaddq_f32(dw_old, gp_inv);
            vst1q_f32(d_weight.add(offset), dw_new);
        }

        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            let g = *grad_base.add(offset);
            let w = *weight.add(offset);
            let pn = *pn_base.add(offset);

            let d_ir = (g * w - pn * coeff) * inv_rms;
            *d_ir_base.add(offset) = d_ir;

            let d_w = g * pn * inv_rms;
            *d_weight.add(offset) += d_w;
        }
    }
}

/// NEON Fused Add + RMS Norm Backward for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_rms_norm_bwd_f64(
    grad: *const f64,
    pre_norm: *const f64,
    weight: *const f64,
    d_input_residual: *mut f64,
    d_weight: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    let chunks = hidden_size / F64_LANES;
    let remainder = hidden_size % F64_LANES;

    for b in 0..batch_size {
        let pn_base = pre_norm.add(b * hidden_size);
        let grad_base = grad.add(b * hidden_size);
        let d_ir_base = d_input_residual.add(b * hidden_size);

        let mut acc_sq = vdupq_n_f64(0.0);
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let pn = vld1q_f64(pn_base.add(offset));
            acc_sq = vfmaq_f64(acc_sq, pn, pn);
        }
        let mut sum_sq = hsum_f64(acc_sq);

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            let pn = *pn_base.add(offset);
            sum_sq += pn * pn;
        }

        let mean_sq = sum_sq / hidden_size as f64;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        let mut dot_acc = vdupq_n_f64(0.0);
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let g = vld1q_f64(grad_base.add(offset));
            let w = vld1q_f64(weight.add(offset));
            let pn = vld1q_f64(pn_base.add(offset));
            let gw = vmulq_f64(g, w);
            dot_acc = vfmaq_f64(dot_acc, gw, pn);
        }
        let mut dot = hsum_f64(dot_acc);

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            let g = *grad_base.add(offset);
            let w = *weight.add(offset);
            let pn = *pn_base.add(offset);
            dot += g * w * pn;
        }

        let coeff = dot * inv_rms / (hidden_size as f64 * (mean_sq + eps));
        let v_inv_rms = vdupq_n_f64(inv_rms);
        let v_coeff = vdupq_n_f64(coeff);

        for i in 0..chunks {
            let offset = i * F64_LANES;
            let g = vld1q_f64(grad_base.add(offset));
            let w = vld1q_f64(weight.add(offset));
            let pn = vld1q_f64(pn_base.add(offset));

            let gw = vmulq_f64(g, w);
            let pn_coeff = vmulq_f64(pn, v_coeff);
            let diff = vsubq_f64(gw, pn_coeff);
            let d_ir = vmulq_f64(diff, v_inv_rms);
            vst1q_f64(d_ir_base.add(offset), d_ir);

            let dw_old = vld1q_f64(d_weight.add(offset));
            let gp = vmulq_f64(g, pn);
            let gp_inv = vmulq_f64(gp, v_inv_rms);
            let dw_new = vaddq_f64(dw_old, gp_inv);
            vst1q_f64(d_weight.add(offset), dw_new);
        }

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            let g = *grad_base.add(offset);
            let w = *weight.add(offset);
            let pn = *pn_base.add(offset);

            let d_ir = (g * w - pn * coeff) * inv_rms;
            *d_ir_base.add(offset) = d_ir;

            let d_w = g * pn * inv_rms;
            *d_weight.add(offset) += d_w;
        }
    }
}
