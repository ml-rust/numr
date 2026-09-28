//! AVX2 + FMA 1D convolution kernels.
//!
//! Vectorises over OUTPUT POSITIONS (see the `driver` module): the weight is a
//! scalar broadcast and the input a contiguous vector load, from the input for
//! `stride == 1` and from the driver's phase buffers for `stride > 1`. Four
//! accumulators covering `4 * LANES` neighbouring outputs keep four
//! independent FMA chains in flight, hiding the 4-5 cycle FMA latency. They
//! use 4 of the 16 vector registers.
//!
//! - f32: 8 lanes per vector, 32 output positions per unrolled iteration
//! - f64: 4 lanes per vector, 16 output positions per unrolled iteration

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::driver::{conv1d_body, conv1d_interior};
use crate::ops::conv_common::Conv1dParams;

/// AVX2 `conv1d` for f32.
///
/// # Safety
/// - All pointers must be valid for the shapes in `params`
/// - CPU must support AVX2 + FMA
#[target_feature(enable = "avx2", enable = "fma")]
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
                lanes = 8,
                zero = _mm256_setzero_ps(),
                splat = _mm256_set1_ps,
                load = _mm256_loadu_ps,
                store = _mm256_storeu_ps,
                add = _mm256_add_ps,
                fma = |acc, x, w| _mm256_fmadd_ps(x, w, acc),
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

/// AVX2 `conv1d` for f64.
///
/// # Safety
/// - All pointers must be valid for the shapes in `params`
/// - CPU must support AVX2 + FMA
#[target_feature(enable = "avx2", enable = "fma")]
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
                lanes = 4,
                zero = _mm256_setzero_pd(),
                splat = _mm256_set1_pd,
                load = _mm256_loadu_pd,
                store = _mm256_storeu_pd,
                add = _mm256_add_pd,
                fma = |acc, x, w| _mm256_fmadd_pd(x, w, acc),
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
