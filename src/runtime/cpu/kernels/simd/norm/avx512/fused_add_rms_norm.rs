//! AVX-512 fused add + RMS normalization kernels (forward and backward)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{F32_LANES, F64_LANES};

/// AVX-512 Fused Add + RMS Normalization for f32
///
/// Computes: output = (input + residual) * rsqrt(mean((input + residual)^2) + eps) * weight
/// Stores intermediate (input + residual) in pre_norm for backward pass.
#[target_feature(enable = "avx512f")]
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

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        let mut acc = _mm512_setzero_ps();
        for c in 0..chunks {
            let offset = row_start + c * F32_LANES;
            let v_in = _mm512_loadu_ps(input.add(offset));
            let v_res = _mm512_loadu_ps(residual.add(offset));
            let pn = _mm512_add_ps(v_in, v_res);
            _mm512_storeu_ps(pre_norm.add(offset), pn);
            acc = _mm512_fmadd_ps(pn, pn, acc);
        }
        let mut sum_sq = _mm512_reduce_add_ps(acc) as f64;

        for i in (chunks * F32_LANES)..hidden_size {
            let pn = *input.add(row_start + i) + *residual.add(row_start + i);
            *pre_norm.add(row_start + i) = pn;
            let pn64 = pn as f64;
            sum_sq += pn64 * pn64;
        }

        let inv_rms = (1.0f64 / (sum_sq / hidden_size as f64 + eps as f64).sqrt()) as f32;
        let v_inv_rms = _mm512_set1_ps(inv_rms);

        for c in 0..chunks {
            let offset = row_start + c * F32_LANES;
            let w_offset = c * F32_LANES;
            let pn = _mm512_loadu_ps(pre_norm.add(offset));
            let v_weight = _mm512_loadu_ps(weight.add(w_offset));
            let v_result = _mm512_mul_ps(_mm512_mul_ps(pn, v_inv_rms), v_weight);
            _mm512_storeu_ps(out.add(offset), v_result);
        }

        for i in (chunks * F32_LANES)..hidden_size {
            let pn = *pre_norm.add(row_start + i);
            let w = *weight.add(i);
            *out.add(row_start + i) = pn * inv_rms * w;
        }
    }
}

/// AVX-512 Fused Add + RMS Normalization for f64
#[target_feature(enable = "avx512f")]
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

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        let mut acc = _mm512_setzero_pd();
        for c in 0..chunks {
            let offset = row_start + c * F64_LANES;
            let v_in = _mm512_loadu_pd(input.add(offset));
            let v_res = _mm512_loadu_pd(residual.add(offset));
            let pn = _mm512_add_pd(v_in, v_res);
            _mm512_storeu_pd(pre_norm.add(offset), pn);
            acc = _mm512_fmadd_pd(pn, pn, acc);
        }
        let mut sum_sq = _mm512_reduce_add_pd(acc);

        for i in (chunks * F64_LANES)..hidden_size {
            let pn = *input.add(row_start + i) + *residual.add(row_start + i);
            *pre_norm.add(row_start + i) = pn;
            sum_sq += pn * pn;
        }

        let inv_rms = 1.0 / (sum_sq / hidden_size as f64 + eps).sqrt();
        let v_inv_rms = _mm512_set1_pd(inv_rms);

        for c in 0..chunks {
            let offset = row_start + c * F64_LANES;
            let w_offset = c * F64_LANES;
            let pn = _mm512_loadu_pd(pre_norm.add(offset));
            let v_weight = _mm512_loadu_pd(weight.add(w_offset));
            let v_result = _mm512_mul_pd(_mm512_mul_pd(pn, v_inv_rms), v_weight);
            _mm512_storeu_pd(out.add(offset), v_result);
        }

        for i in (chunks * F64_LANES)..hidden_size {
            let pn = *pre_norm.add(row_start + i);
            let w = *weight.add(i);
            *out.add(row_start + i) = pn * inv_rms * w;
        }
    }
}

/// AVX-512 Fused Add + RMS Norm Backward for f32
#[target_feature(enable = "avx512f")]
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

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        let mut acc_sq = _mm512_setzero_ps();
        for c in 0..chunks {
            let offset = row_start + c * F32_LANES;
            let pn = _mm512_loadu_ps(pre_norm.add(offset));
            acc_sq = _mm512_fmadd_ps(pn, pn, acc_sq);
        }
        let mut sum_sq = _mm512_reduce_add_ps(acc_sq);

        for i in (chunks * F32_LANES)..hidden_size {
            let pn = *pre_norm.add(row_start + i);
            sum_sq += pn * pn;
        }

        let mean_sq = sum_sq / hidden_size as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        let mut dot_acc = _mm512_setzero_ps();
        for c in 0..chunks {
            let offset = row_start + c * F32_LANES;
            let w_offset = c * F32_LANES;
            let g = _mm512_loadu_ps(grad.add(offset));
            let w = _mm512_loadu_ps(weight.add(w_offset));
            let pn = _mm512_loadu_ps(pre_norm.add(offset));
            let gw = _mm512_mul_ps(g, w);
            dot_acc = _mm512_fmadd_ps(gw, pn, dot_acc);
        }
        let mut dot = _mm512_reduce_add_ps(dot_acc);

        for i in (chunks * F32_LANES)..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i);
            dot += g * w * pn;
        }

        let coeff = dot * inv_rms / (hidden_size as f32 * (mean_sq + eps));
        let v_inv_rms = _mm512_set1_ps(inv_rms);
        let v_coeff = _mm512_set1_ps(coeff);

        for c in 0..chunks {
            let offset = row_start + c * F32_LANES;
            let w_offset = c * F32_LANES;
            let g = _mm512_loadu_ps(grad.add(offset));
            let w = _mm512_loadu_ps(weight.add(w_offset));
            let pn = _mm512_loadu_ps(pre_norm.add(offset));

            let gw = _mm512_mul_ps(g, w);
            let pn_coeff = _mm512_mul_ps(pn, v_coeff);
            let diff = _mm512_sub_ps(gw, pn_coeff);
            let d_ir = _mm512_mul_ps(diff, v_inv_rms);
            _mm512_storeu_ps(d_input_residual.add(offset), d_ir);

            let dw_old = _mm512_loadu_ps(d_weight.add(w_offset));
            let gp = _mm512_mul_ps(g, pn);
            let gp_inv = _mm512_mul_ps(gp, v_inv_rms);
            let dw_new = _mm512_add_ps(dw_old, gp_inv);
            _mm512_storeu_ps(d_weight.add(w_offset), dw_new);
        }

        for i in (chunks * F32_LANES)..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i);

            let d_ir = (g * w - pn * coeff) * inv_rms;
            *d_input_residual.add(row_start + i) = d_ir;

            let d_w = g * pn * inv_rms;
            *d_weight.add(i) += d_w;
        }
    }
}

/// AVX-512 Fused Add + RMS Norm Backward for f64
#[target_feature(enable = "avx512f")]
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

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        let mut acc_sq = _mm512_setzero_pd();
        for c in 0..chunks {
            let offset = row_start + c * F64_LANES;
            let pn = _mm512_loadu_pd(pre_norm.add(offset));
            acc_sq = _mm512_fmadd_pd(pn, pn, acc_sq);
        }
        let mut sum_sq = _mm512_reduce_add_pd(acc_sq);

        for i in (chunks * F64_LANES)..hidden_size {
            let pn = *pre_norm.add(row_start + i);
            sum_sq += pn * pn;
        }

        let mean_sq = sum_sq / hidden_size as f64;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        let mut dot_acc = _mm512_setzero_pd();
        for c in 0..chunks {
            let offset = row_start + c * F64_LANES;
            let w_offset = c * F64_LANES;
            let g = _mm512_loadu_pd(grad.add(offset));
            let w = _mm512_loadu_pd(weight.add(w_offset));
            let pn = _mm512_loadu_pd(pre_norm.add(offset));
            let gw = _mm512_mul_pd(g, w);
            dot_acc = _mm512_fmadd_pd(gw, pn, dot_acc);
        }
        let mut dot = _mm512_reduce_add_pd(dot_acc);

        for i in (chunks * F64_LANES)..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i);
            dot += g * w * pn;
        }

        let coeff = dot * inv_rms / (hidden_size as f64 * (mean_sq + eps));
        let v_inv_rms = _mm512_set1_pd(inv_rms);
        let v_coeff = _mm512_set1_pd(coeff);

        for c in 0..chunks {
            let offset = row_start + c * F64_LANES;
            let w_offset = c * F64_LANES;
            let g = _mm512_loadu_pd(grad.add(offset));
            let w = _mm512_loadu_pd(weight.add(w_offset));
            let pn = _mm512_loadu_pd(pre_norm.add(offset));

            let gw = _mm512_mul_pd(g, w);
            let pn_coeff = _mm512_mul_pd(pn, v_coeff);
            let diff = _mm512_sub_pd(gw, pn_coeff);
            let d_ir = _mm512_mul_pd(diff, v_inv_rms);
            _mm512_storeu_pd(d_input_residual.add(offset), d_ir);

            let dw_old = _mm512_loadu_pd(d_weight.add(w_offset));
            let gp = _mm512_mul_pd(g, pn);
            let gp_inv = _mm512_mul_pd(gp, v_inv_rms);
            let dw_new = _mm512_add_pd(dw_old, gp_inv);
            _mm512_storeu_pd(d_weight.add(w_offset), dw_new);
        }

        for i in (chunks * F64_LANES)..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i);

            let d_ir = (g * w - pn * coeff) * inv_rms;
            *d_input_residual.add(row_start + i) = d_ir;

            let d_w = g * pn * inv_rms;
            *d_weight.add(i) += d_w;
        }
    }
}
