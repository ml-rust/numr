//! NEON softmax kernels for ARM64
//!
//! Provides vectorized softmax operation using 128-bit NEON registers.
//!
//! softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
//!
//! # SIMD Strategy
//!
//! 1. SIMD max-reduce to find maximum for numerical stability
//! 2. SIMD exp computation with shifted values
//! 3. SIMD sum-reduce for normalization factor
//! 4. SIMD multiply by inverse sum

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::super::math::aarch64::neon::{
    exp_f32, exp_f64, hmax_f32, hmax_f64, hsum_f32, hsum_f64,
};

const F32_LANES: usize = 4;
const F64_LANES: usize = 2;

/// NEON softmax for f32
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

        // Phase 1: Find max (for numerical stability)
        let mut max_acc = vdupq_n_f32(f32::NEG_INFINITY);
        for i in 0..chunks {
            let v = vld1q_f32(base.add(i * F32_LANES));
            max_acc = vmaxq_f32(max_acc, v);
        }
        let mut max_val = hmax_f32(max_acc);

        // Scalar tail for max
        for i in 0..remainder {
            let val = *base.add(chunks * F32_LANES + i);
            if val > max_val {
                max_val = val;
            }
        }

        let v_max = vdupq_n_f32(max_val);

        // Phase 2: Compute exp(x - max) and sum
        let mut sum_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let v = vld1q_f32(base.add(offset));
            let shifted = vsubq_f32(v, v_max);
            let exp_v = exp_f32(shifted);
            vst1q_f32(out_base.add(offset), exp_v);
            sum_acc = vaddq_f32(sum_acc, exp_v);
        }
        let mut sum = hsum_f32(sum_acc);

        // Scalar tail for exp and sum
        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            let val = *base.add(offset);
            let exp_val = (val - max_val).exp();
            *out_base.add(offset) = exp_val;
            sum += exp_val;
        }

        // Phase 3: Normalize by sum
        let inv_sum = vdupq_n_f32(1.0 / sum);
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let v = vld1q_f32(out_base.add(offset));
            vst1q_f32(out_base.add(offset), vmulq_f32(v, inv_sum));
        }

        // Scalar tail for normalization
        let scalar_inv_sum = 1.0 / sum;
        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            *out_base.add(offset) *= scalar_inv_sum;
        }
    }
}

/// NEON softmax for f64
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

        // Phase 1: Find max
        let mut max_acc = vdupq_n_f64(f64::NEG_INFINITY);
        for i in 0..chunks {
            let v = vld1q_f64(base.add(i * F64_LANES));
            max_acc = vmaxq_f64(max_acc, v);
        }
        let mut max_val = hmax_f64(max_acc);

        for i in 0..remainder {
            let val = *base.add(chunks * F64_LANES + i);
            if val > max_val {
                max_val = val;
            }
        }

        let v_max = vdupq_n_f64(max_val);

        // Phase 2: Compute exp(x - max) and sum
        let mut sum_acc = vdupq_n_f64(0.0);
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let v = vld1q_f64(base.add(offset));
            let shifted = vsubq_f64(v, v_max);
            let exp_v = exp_f64(shifted);
            vst1q_f64(out_base.add(offset), exp_v);
            sum_acc = vaddq_f64(sum_acc, exp_v);
        }
        let mut sum = hsum_f64(sum_acc);

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            let val = *base.add(offset);
            let exp_val = (val - max_val).exp();
            *out_base.add(offset) = exp_val;
            sum += exp_val;
        }

        // Phase 3: Normalize
        let inv_sum = vdupq_n_f64(1.0 / sum);
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let v = vld1q_f64(out_base.add(offset));
            vst1q_f64(out_base.add(offset), vmulq_f64(v, inv_sum));
        }

        let scalar_inv_sum = 1.0 / sum;
        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            *out_base.add(offset) *= scalar_inv_sum;
        }
    }
}
