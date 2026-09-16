//! Runtime `SimdLevel` dispatch for `conv2d`.
//!
//! # SIMD Strategy
//!
//! `conv2d` vectorizes the inner loop over INPUT CHANNELS:
//! - AVX2: Process 8 f32 channels or 4 f64 channels per iteration
//! - AVX-512: Process 16 f32 channels or 8 f64 channels per iteration
//!
//! and falls back to scalar for convolutions with few input channels (< 8).
//!
//! `conv1d`, `conv_transpose1d` and `depthwise_conv2d` vectorize over OUTPUT
//! POSITIONS instead. Channels are `length` elements apart in a `(B, C, L)`
//! layout, so channel vectorization cannot use a vector load at all, and it
//! degenerates entirely for depthwise shapes (`c_in_per_group == 1`). Output
//! positions are contiguous, which makes the weight a scalar broadcast and the
//! input a contiguous load. Those kernels therefore gate on the OUTPUT SIZE, not
//! the channel count.

use super::scalar::{conv2d_scalar_f32, conv2d_scalar_f64};
use super::threshold::{SIMD_THRESHOLD_F32, SIMD_THRESHOLD_F64};
use crate::ops::conv_common::Conv2dParams;
use crate::runtime::cpu::kernels::simd::{SimdLevel, detect_simd};

#[cfg(target_arch = "aarch64")]
use super::aarch64::neon;
#[cfg(target_arch = "x86_64")]
use super::{avx2, avx512};

/// SIMD conv2d for f32
///
/// # Safety
/// - All pointers must be valid and properly aligned
/// - Arrays must have sufficient size for the operation
#[inline]
pub unsafe fn conv2d_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: Conv2dParams,
) {
    let level = detect_simd();
    let c_in_per_group = params.c_in / params.groups;

    if c_in_per_group < SIMD_THRESHOLD_F32 || level == SimdLevel::Scalar {
        conv2d_scalar_f32(input, weight, bias, output, params);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::conv2d_f32(input, weight, bias, output, params),
        SimdLevel::Avx2Fma => avx2::conv2d_f32(input, weight, bias, output, params),
        _ => conv2d_scalar_f32(input, weight, bias, output, params),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            neon::conv2d_f32(input, weight, bias, output, params)
        }
        _ => conv2d_scalar_f32(input, weight, bias, output, params),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    conv2d_scalar_f32(input, weight, bias, output, params);
}

/// SIMD conv2d for f64
///
/// # Safety
/// - All pointers must be valid and properly aligned
/// - Arrays must have sufficient size for the operation
#[inline]
pub unsafe fn conv2d_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: Conv2dParams,
) {
    let level = detect_simd();
    let c_in_per_group = params.c_in / params.groups;

    if c_in_per_group < SIMD_THRESHOLD_F64 || level == SimdLevel::Scalar {
        conv2d_scalar_f64(input, weight, bias, output, params);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::conv2d_f64(input, weight, bias, output, params),
        SimdLevel::Avx2Fma => avx2::conv2d_f64(input, weight, bias, output, params),
        _ => conv2d_scalar_f64(input, weight, bias, output, params),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            neon::conv2d_f64(input, weight, bias, output, params)
        }
        _ => conv2d_scalar_f64(input, weight, bias, output, params),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    conv2d_scalar_f64(input, weight, bias, output, params);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::ops::PaddingMode;
    use crate::ops::conv_common::validate_conv2d;

    #[test]
    fn test_conv2d_simd_matches_scalar() {
        // Input: (1, 16, 8, 8) - 16 channels to trigger SIMD
        let c_in = 16;
        let (h, w) = (8, 8);
        let c_out = 4;
        let (kh, kw) = (3, 3);

        let input: Vec<f32> = (0..(c_in * h * w)).map(|x| (x as f32) * 0.01).collect();
        let weight: Vec<f32> = (0..(c_out * c_in * kh * kw))
            .map(|x| (x as f32) * 0.001 - 0.2)
            .collect();

        let params = validate_conv2d(
            &[1, c_in, h, w],
            &[c_out, c_in, kh, kw],
            None,
            (1, 1),
            PaddingMode::Valid,
            (1, 1),
            1,
            DType::F32,
            DType::F32,
            None,
        )
        .unwrap();

        let output_len = c_out * params.output_h * params.output_w;
        let mut out_simd = vec![0.0f32; output_len];
        let mut out_scalar = vec![0.0f32; output_len];

        unsafe {
            conv2d_f32(
                input.as_ptr(),
                weight.as_ptr(),
                None,
                out_simd.as_mut_ptr(),
                params,
            );
            conv2d_scalar_f32(
                input.as_ptr(),
                weight.as_ptr(),
                None,
                out_scalar.as_mut_ptr(),
                params,
            );
        }

        for i in 0..output_len {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            let rel_err = if out_scalar[i].abs() > 1e-6 {
                diff / out_scalar[i].abs()
            } else {
                diff
            };
            assert!(
                rel_err < 1e-4,
                "conv2d mismatch at {}: SIMD={} scalar={} (rel_err={})",
                i,
                out_simd[i],
                out_scalar[i],
                rel_err
            );
        }
    }
}
