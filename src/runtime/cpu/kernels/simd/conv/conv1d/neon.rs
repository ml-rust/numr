//! NEON 1D convolution kernels for AArch64.
//!
//! Vectorises over OUTPUT POSITIONS (see the `driver` module): the weight is a
//! scalar broadcast and the input a contiguous vector load, from the input for
//! `stride == 1` and from the driver's phase buffers for `stride > 1`. Four
//! accumulators covering `4 * LANES` neighbouring outputs keep four
//! independent FMA chains in flight.
//!
//! - f32: 4 lanes per vector, 16 output positions per unrolled iteration
//! - f64: 2 lanes per vector, 8 output positions per unrolled iteration

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::driver::{conv1d_body, conv1d_interior};
use crate::ops::conv_common::Conv1dParams;

/// NEON `conv1d` for f32.
///
/// # Safety
/// - All pointers must be valid for the shapes in `params`
/// - CPU must support NEON
#[target_feature(enable = "neon")]
pub unsafe fn conv1d_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: Conv1dParams,
) {
    conv1d_body!(
        f32,
        input,
        weight,
        bias,
        output,
        params,
        |op, ip, wp, n, nic, bv, rs, taps| {
            conv1d_interior!(
                f32,
                lanes = 4,
                zero = vdupq_n_f32(0.0),
                splat = vdupq_n_f32,
                load = vld1q_f32,
                store = vst1q_f32,
                add = vaddq_f32,
                fma = |acc, x, w| vfmaq_f32(acc, x, w),
                op,
                ip,
                wp,
                n,
                nic,
                bv,
                rs,
                taps,
                params.kernel_size
            );
        }
    );
}

/// NEON `conv1d` for f64.
///
/// # Safety
/// - All pointers must be valid for the shapes in `params`
/// - CPU must support NEON
#[target_feature(enable = "neon")]
pub unsafe fn conv1d_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: Conv1dParams,
) {
    conv1d_body!(
        f64,
        input,
        weight,
        bias,
        output,
        params,
        |op, ip, wp, n, nic, bv, rs, taps| {
            conv1d_interior!(
                f64,
                lanes = 2,
                zero = vdupq_n_f64(0.0),
                splat = vdupq_n_f64,
                load = vld1q_f64,
                store = vst1q_f64,
                add = vaddq_f64,
                fma = |acc, x, w| vfmaq_f64(acc, x, w),
                op,
                ip,
                wp,
                n,
                nic,
                bv,
                rs,
                taps,
                params.kernel_size
            );
        }
    );
}
