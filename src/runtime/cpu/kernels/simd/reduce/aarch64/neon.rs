//! NEON reduction kernels for ARM64
//!
//! Provides vectorized reduction operations using 128-bit NEON registers.
//!
//! # Supported Operations
//!
//! - Sum: Accumulate all elements
//! - Max: Find maximum element
//! - Min: Find minimum element
//! - Prod: Product of all elements
//!
//! # SIMD Strategy
//!
//! 1. Process 4 f32 / 2 f64 elements per iteration
//! 2. Accumulate into NEON registers
//! 3. Horizontal reduction at the end
//! 4. Scalar tail for remaining elements

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::super::math::aarch64::neon::{
    hmax_f32, hmax_f64, hmin_f32, hmin_f64, hsum_f32, hsum_f64,
};
use crate::ops::ReduceOp;

const F32_LANES: usize = 4;
const F64_LANES: usize = 2;

/// NEON reduction for f32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a` must point to `reduce_size * outer_size` valid f32 elements
/// - `out` must point to `outer_size` valid f32 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn reduce_f32(
    op: ReduceOp,
    a: *const f32,
    out: *mut f32,
    reduce_size: usize,
    outer_size: usize,
) {
    match op {
        ReduceOp::Sum => reduce_sum_f32(a, out, reduce_size, outer_size),
        ReduceOp::Max => reduce_max_f32(a, out, reduce_size, outer_size),
        ReduceOp::Min => reduce_min_f32(a, out, reduce_size, outer_size),
        ReduceOp::Prod => reduce_prod_f32(a, out, reduce_size, outer_size),
        ReduceOp::Mean => {
            reduce_sum_f32(a, out, reduce_size, outer_size);
            let scale = 1.0 / reduce_size as f32;
            let v_scale = vdupq_n_f32(scale);
            let chunks = outer_size / F32_LANES;
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let v = vld1q_f32(out.add(offset));
                vst1q_f32(out.add(offset), vmulq_f32(v, v_scale));
            }
            for i in (chunks * F32_LANES)..outer_size {
                *out.add(i) *= scale;
            }
        }
        ReduceOp::All | ReduceOp::Any => {
            // Boolean operations - use scalar
            super::super::reduce_scalar_f32(op, a, out, reduce_size, outer_size);
        }
    }
}

/// NEON reduction for f64
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a` must point to `reduce_size * outer_size` valid f64 elements
/// - `out` must point to `outer_size` valid f64 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn reduce_f64(
    op: ReduceOp,
    a: *const f64,
    out: *mut f64,
    reduce_size: usize,
    outer_size: usize,
) {
    match op {
        ReduceOp::Sum => reduce_sum_f64(a, out, reduce_size, outer_size),
        ReduceOp::Max => reduce_max_f64(a, out, reduce_size, outer_size),
        ReduceOp::Min => reduce_min_f64(a, out, reduce_size, outer_size),
        ReduceOp::Prod => reduce_prod_f64(a, out, reduce_size, outer_size),
        ReduceOp::Mean => {
            reduce_sum_f64(a, out, reduce_size, outer_size);
            let scale = 1.0 / reduce_size as f64;
            let v_scale = vdupq_n_f64(scale);
            let chunks = outer_size / F64_LANES;
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let v = vld1q_f64(out.add(offset));
                vst1q_f64(out.add(offset), vmulq_f64(v, v_scale));
            }
            for i in (chunks * F64_LANES)..outer_size {
                *out.add(i) *= scale;
            }
        }
        ReduceOp::All | ReduceOp::Any => {
            super::super::reduce_scalar_f64(op, a, out, reduce_size, outer_size);
        }
    }
}

// ============================================================================
// Sum reductions
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn reduce_sum_f32(a: *const f32, out: *mut f32, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F32_LANES;
    let remainder = reduce_size % F32_LANES;

    for o in 0..outer_size {
        let base = a.add(o * reduce_size);
        let mut acc = vdupq_n_f32(0.0);

        // SIMD accumulation
        for i in 0..chunks {
            let v = vld1q_f32(base.add(i * F32_LANES));
            acc = vaddq_f32(acc, v);
        }

        // Horizontal sum
        let mut sum = hsum_f32(acc);

        // Scalar tail
        for i in 0..remainder {
            sum += *base.add(chunks * F32_LANES + i);
        }

        *out.add(o) = sum;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn reduce_sum_f64(a: *const f64, out: *mut f64, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F64_LANES;
    let remainder = reduce_size % F64_LANES;

    for o in 0..outer_size {
        let base = a.add(o * reduce_size);
        let mut acc = vdupq_n_f64(0.0);

        for i in 0..chunks {
            let v = vld1q_f64(base.add(i * F64_LANES));
            acc = vaddq_f64(acc, v);
        }

        let mut sum = hsum_f64(acc);

        for i in 0..remainder {
            sum += *base.add(chunks * F64_LANES + i);
        }

        *out.add(o) = sum;
    }
}

// ============================================================================
// Max reductions
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn reduce_max_f32(a: *const f32, out: *mut f32, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F32_LANES;
    let remainder = reduce_size % F32_LANES;

    for o in 0..outer_size {
        let base = a.add(o * reduce_size);

        // Initialize with negative infinity
        let mut acc = vdupq_n_f32(f32::NEG_INFINITY);

        for i in 0..chunks {
            let v = vld1q_f32(base.add(i * F32_LANES));
            acc = vmaxq_f32(acc, v);
        }

        let mut max_val = hmax_f32(acc);

        for i in 0..remainder {
            let val = *base.add(chunks * F32_LANES + i);
            if val > max_val {
                max_val = val;
            }
        }

        *out.add(o) = max_val;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn reduce_max_f64(a: *const f64, out: *mut f64, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F64_LANES;
    let remainder = reduce_size % F64_LANES;

    for o in 0..outer_size {
        let base = a.add(o * reduce_size);
        let mut acc = vdupq_n_f64(f64::NEG_INFINITY);

        for i in 0..chunks {
            let v = vld1q_f64(base.add(i * F64_LANES));
            acc = vmaxq_f64(acc, v);
        }

        let mut max_val = hmax_f64(acc);

        for i in 0..remainder {
            let val = *base.add(chunks * F64_LANES + i);
            if val > max_val {
                max_val = val;
            }
        }

        *out.add(o) = max_val;
    }
}

// ============================================================================
// Min reductions
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn reduce_min_f32(a: *const f32, out: *mut f32, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F32_LANES;
    let remainder = reduce_size % F32_LANES;

    for o in 0..outer_size {
        let base = a.add(o * reduce_size);
        let mut acc = vdupq_n_f32(f32::INFINITY);

        for i in 0..chunks {
            let v = vld1q_f32(base.add(i * F32_LANES));
            acc = vminq_f32(acc, v);
        }

        let mut min_val = hmin_f32(acc);

        for i in 0..remainder {
            let val = *base.add(chunks * F32_LANES + i);
            if val < min_val {
                min_val = val;
            }
        }

        *out.add(o) = min_val;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn reduce_min_f64(a: *const f64, out: *mut f64, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F64_LANES;
    let remainder = reduce_size % F64_LANES;

    for o in 0..outer_size {
        let base = a.add(o * reduce_size);
        let mut acc = vdupq_n_f64(f64::INFINITY);

        for i in 0..chunks {
            let v = vld1q_f64(base.add(i * F64_LANES));
            acc = vminq_f64(acc, v);
        }

        let mut min_val = hmin_f64(acc);

        for i in 0..remainder {
            let val = *base.add(chunks * F64_LANES + i);
            if val < min_val {
                min_val = val;
            }
        }

        *out.add(o) = min_val;
    }
}

// ============================================================================
// Product reductions
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn reduce_prod_f32(a: *const f32, out: *mut f32, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F32_LANES;
    let remainder = reduce_size % F32_LANES;

    for o in 0..outer_size {
        let base = a.add(o * reduce_size);
        let mut acc = vdupq_n_f32(1.0);

        for i in 0..chunks {
            let v = vld1q_f32(base.add(i * F32_LANES));
            acc = vmulq_f32(acc, v);
        }

        // Horizontal product (no native instruction, extract and multiply)
        let mut prod = vgetq_lane_f32::<0>(acc)
            * vgetq_lane_f32::<1>(acc)
            * vgetq_lane_f32::<2>(acc)
            * vgetq_lane_f32::<3>(acc);

        for i in 0..remainder {
            prod *= *base.add(chunks * F32_LANES + i);
        }

        *out.add(o) = prod;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn reduce_prod_f64(a: *const f64, out: *mut f64, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F64_LANES;
    let remainder = reduce_size % F64_LANES;

    for o in 0..outer_size {
        let base = a.add(o * reduce_size);
        let mut acc = vdupq_n_f64(1.0);

        for i in 0..chunks {
            let v = vld1q_f64(base.add(i * F64_LANES));
            acc = vmulq_f64(acc, v);
        }

        let mut prod = vgetq_lane_f64::<0>(acc) * vgetq_lane_f64::<1>(acc);

        for i in 0..remainder {
            prod *= *base.add(chunks * F64_LANES + i);
        }

        *out.add(o) = prod;
    }
}
