//! NEON softmax kernels for ARM64 using online algorithm (2-pass).
//!
//! Pass 1: Online SIMD max + sum (single read of input)
//! Pass 2: Compute exp(x - max) / sum and write output (one read + one write)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::super::math::aarch64::neon::{
    exp_f32, exp_f64, hmax_f32, hmax_f64, hsum_f32, hsum_f64,
};

const F32_LANES: usize = 4;
const F64_LANES: usize = 2;

/// NEON softmax for f32 using online algorithm.
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a` and `out` must point to `outer_size * dim_size` valid f32 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn softmax_f32(a: *const f32, out: *mut f32, outer_size: usize, dim_size: usize) {
    let chunks = dim_size / F32_LANES;
    let remainder = dim_size % F32_LANES;

    for o in 0..outer_size {
        let base = a.add(o * dim_size);
        let out_base = out.add(o * dim_size);

        // Pass 1: Online max + sum
        let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
        let mut sum_vec = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let v = vld1q_f32(base.add(i * F32_LANES));

            let old_max = max_vec;
            max_vec = vmaxq_f32(max_vec, v);

            // Rescale previous sum
            let rescale = exp_f32(vsubq_f32(old_max, max_vec));
            sum_vec = vmulq_f32(sum_vec, rescale);

            // Add new contributions
            let exp_v = exp_f32(vsubq_f32(v, max_vec));
            sum_vec = vaddq_f32(sum_vec, exp_v);
        }

        // Horizontal reduce to get per-lane max, then reconcile with scalar tail
        let mut max_val = hmax_f32(max_vec);

        // Scalar tail (online)
        let mut tail_sum = 0.0f32;
        for i in 0..remainder {
            let val = *base.add(chunks * F32_LANES + i);
            if val > max_val {
                tail_sum = tail_sum * (max_val - val).exp() + 1.0;
                max_val = val;
            } else {
                tail_sum += (val - max_val).exp();
            }
        }

        // Reconcile SIMD sum with global max
        let v_global_max = vdupq_n_f32(max_val);
        let rescale = exp_f32(vsubq_f32(max_vec, v_global_max));
        let rescaled_sum = vmulq_f32(sum_vec, rescale);
        let sum = hsum_f32(rescaled_sum) + tail_sum;

        // Pass 2: exp(x - max) / sum
        let v_max = vdupq_n_f32(max_val);
        let inv_sum_vec = vdupq_n_f32(1.0 / sum);

        for i in 0..chunks {
            let offset = i * F32_LANES;
            let v = vld1q_f32(base.add(offset));
            let shifted = vsubq_f32(v, v_max);
            let normalized = vmulq_f32(exp_f32(shifted), inv_sum_vec);
            vst1q_f32(out_base.add(offset), normalized);
        }

        let scalar_inv_sum = 1.0 / sum;
        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            let val = *base.add(offset);
            *out_base.add(offset) = (val - max_val).exp() * scalar_inv_sum;
        }
    }
}

/// NEON softmax for f64 using online algorithm.
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a` and `out` must point to `outer_size * dim_size` valid f64 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn softmax_f64(a: *const f64, out: *mut f64, outer_size: usize, dim_size: usize) {
    let chunks = dim_size / F64_LANES;
    let remainder = dim_size % F64_LANES;

    for o in 0..outer_size {
        let base = a.add(o * dim_size);
        let out_base = out.add(o * dim_size);

        // Pass 1: Online max + sum
        let mut max_vec = vdupq_n_f64(f64::NEG_INFINITY);
        let mut sum_vec = vdupq_n_f64(0.0);

        for i in 0..chunks {
            let v = vld1q_f64(base.add(i * F64_LANES));

            let old_max = max_vec;
            max_vec = vmaxq_f64(max_vec, v);

            let rescale = exp_f64(vsubq_f64(old_max, max_vec));
            sum_vec = vmulq_f64(sum_vec, rescale);

            let exp_v = exp_f64(vsubq_f64(v, max_vec));
            sum_vec = vaddq_f64(sum_vec, exp_v);
        }

        let mut max_val = hmax_f64(max_vec);

        let mut tail_sum = 0.0f64;
        for i in 0..remainder {
            let val = *base.add(chunks * F64_LANES + i);
            if val > max_val {
                tail_sum = tail_sum * (max_val - val).exp() + 1.0;
                max_val = val;
            } else {
                tail_sum += (val - max_val).exp();
            }
        }

        // Reconcile SIMD sum with global max
        let v_global_max = vdupq_n_f64(max_val);
        let rescale = exp_f64(vsubq_f64(max_vec, v_global_max));
        let rescaled_sum = vmulq_f64(sum_vec, rescale);
        let sum = hsum_f64(rescaled_sum) + tail_sum;

        // Pass 2: exp(x - max) / sum
        let v_max = vdupq_n_f64(max_val);
        let inv_sum_vec = vdupq_n_f64(1.0 / sum);

        for i in 0..chunks {
            let offset = i * F64_LANES;
            let v = vld1q_f64(base.add(offset));
            let shifted = vsubq_f64(v, v_max);
            let normalized = vmulq_f64(exp_f64(shifted), inv_sum_vec);
            vst1q_f64(out_base.add(offset), normalized);
        }

        let scalar_inv_sum = 1.0 / sum;
        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            let val = *base.add(offset);
            *out_base.add(offset) = (val - max_val).exp() * scalar_inv_sum;
        }
    }
}
