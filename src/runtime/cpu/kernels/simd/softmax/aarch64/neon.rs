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
            // Guard: when old_max == max_vec == -inf, exp(-inf-(-inf)) = NaN.
            // Use mask to zero out -inf lanes (their sum contribution is 0).
            let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);
            let valid_old = vmvnq_u32(vceqq_f32(old_max, neg_inf)); // != -inf
            let rescale = exp_f32(vsubq_f32(old_max, max_vec));
            let rescale =
                vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(rescale), valid_old));
            sum_vec = vmulq_f32(sum_vec, rescale);

            // Add new contributions
            let valid_new = vmvnq_u32(vceqq_f32(max_vec, neg_inf)); // != -inf
            let exp_v = exp_f32(vsubq_f32(v, max_vec));
            let exp_v = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(exp_v), valid_new));
            sum_vec = vaddq_f32(sum_vec, exp_v);
        }

        // Horizontal reduce to get per-lane max, then reconcile with scalar tail
        let mut max_val = hmax_f32(max_vec);

        // Scalar tail (online)
        let mut tail_sum = 0.0f32;
        for i in 0..remainder {
            let val = *base.add(chunks * F32_LANES + i);
            if val > max_val {
                let rescale = if max_val == f32::NEG_INFINITY {
                    0.0
                } else {
                    (max_val - val).exp()
                };
                tail_sum = tail_sum * rescale + 1.0;
                max_val = val;
            } else if val == f32::NEG_INFINITY {
                // skip
            } else {
                tail_sum += (val - max_val).exp();
            }
        }

        // Reconcile SIMD sum with global max
        // Guard -inf lanes to avoid NaN from exp(-inf - (-inf))
        let neg_inf = vdupq_n_f32(f32::NEG_INFINITY);
        let valid_mask = vmvnq_u32(vceqq_f32(max_vec, neg_inf));
        let v_global_max = vdupq_n_f32(max_val);
        let rescale = exp_f32(vsubq_f32(max_vec, v_global_max));
        let rescale = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(rescale), valid_mask));
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

            // Guard -inf lanes
            let neg_inf = vdupq_n_f64(f64::NEG_INFINITY);
            let valid_old = veorq_u64(vceqq_f64(old_max, neg_inf), vdupq_n_u64(!0));
            let rescale = exp_f64(vsubq_f64(old_max, max_vec));
            let rescale =
                vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(rescale), valid_old));
            sum_vec = vmulq_f64(sum_vec, rescale);

            let valid_new = veorq_u64(vceqq_f64(max_vec, neg_inf), vdupq_n_u64(!0));
            let exp_v = exp_f64(vsubq_f64(v, max_vec));
            let exp_v = vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(exp_v), valid_new));
            sum_vec = vaddq_f64(sum_vec, exp_v);
        }

        let mut max_val = hmax_f64(max_vec);

        let mut tail_sum = 0.0f64;
        for i in 0..remainder {
            let val = *base.add(chunks * F64_LANES + i);
            if val > max_val {
                let rescale = if max_val == f64::NEG_INFINITY {
                    0.0
                } else {
                    (max_val - val).exp()
                };
                tail_sum = tail_sum * rescale + 1.0;
                max_val = val;
            } else if val == f64::NEG_INFINITY {
                // skip
            } else {
                tail_sum += (val - max_val).exp();
            }
        }

        // Reconcile SIMD sum with global max
        let neg_inf = vdupq_n_f64(f64::NEG_INFINITY);
        let valid_mask = veorq_u64(vceqq_f64(max_vec, neg_inf), vdupq_n_u64(!0));
        let v_global_max = vdupq_n_f64(max_val);
        let rescale = exp_f64(vsubq_f64(max_vec, v_global_max));
        let rescale = vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(rescale), valid_mask));
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
