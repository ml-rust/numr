//! SIMD-accelerated softmax backward operation.
//!
//! Computes: d_input[i] = output[i] * (grad[i] - dot)
//! where dot = sum(grad * output) along the softmax dimension.
//!
//! Fused 2-pass kernel:
//! - Pass 1: SIMD dot product (grad * output, reduced to scalar)
//! - Pass 2: SIMD elementwise output * (grad - dot)

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

use super::{SimdLevel, detect_simd};

/// Minimum dimension size to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

/// SIMD softmax backward for f32
///
/// # Safety
/// - `grad`, `output`, `d_input` must point to `outer_size * dim_size` elements
#[inline]
pub unsafe fn softmax_bwd_f32(
    grad: *const f32,
    output: *const f32,
    d_input: *mut f32,
    outer_size: usize,
    dim_size: usize,
) {
    let level = detect_simd();

    if dim_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        softmax_bwd_scalar_f32(grad, output, d_input, outer_size, dim_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::softmax_bwd_f32(grad, output, d_input, outer_size, dim_size),
        SimdLevel::Avx2Fma => avx2::softmax_bwd_f32(grad, output, d_input, outer_size, dim_size),
        _ => softmax_bwd_scalar_f32(grad, output, d_input, outer_size, dim_size),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::softmax_bwd_f32(grad, output, d_input, outer_size, dim_size)
        }
        _ => softmax_bwd_scalar_f32(grad, output, d_input, outer_size, dim_size),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    softmax_bwd_scalar_f32(grad, output, d_input, outer_size, dim_size);
}

/// SIMD softmax backward for f64
///
/// # Safety
/// - `grad`, `output`, `d_input` must point to `outer_size * dim_size` elements
#[inline]
pub unsafe fn softmax_bwd_f64(
    grad: *const f64,
    output: *const f64,
    d_input: *mut f64,
    outer_size: usize,
    dim_size: usize,
) {
    let level = detect_simd();

    if dim_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        softmax_bwd_scalar_f64(grad, output, d_input, outer_size, dim_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::softmax_bwd_f64(grad, output, d_input, outer_size, dim_size),
        SimdLevel::Avx2Fma => avx2::softmax_bwd_f64(grad, output, d_input, outer_size, dim_size),
        _ => softmax_bwd_scalar_f64(grad, output, d_input, outer_size, dim_size),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::softmax_bwd_f64(grad, output, d_input, outer_size, dim_size)
        }
        _ => softmax_bwd_scalar_f64(grad, output, d_input, outer_size, dim_size),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    softmax_bwd_scalar_f64(grad, output, d_input, outer_size, dim_size);
}

// ============================================================================
// Scalar fallbacks
// ============================================================================

/// Scalar softmax backward for f32
#[inline]
pub unsafe fn softmax_bwd_scalar_f32(
    grad: *const f32,
    output: *const f32,
    d_input: *mut f32,
    outer_size: usize,
    dim_size: usize,
) {
    for o in 0..outer_size {
        let base = o * dim_size;

        // Pass 1: dot = sum(grad * output)
        let mut dot = 0.0f32;
        for d in 0..dim_size {
            dot += *grad.add(base + d) * *output.add(base + d);
        }

        // Pass 2: d_input = output * (grad - dot)
        for d in 0..dim_size {
            let idx = base + d;
            *d_input.add(idx) = *output.add(idx) * (*grad.add(idx) - dot);
        }
    }
}

/// Scalar softmax backward for f64
#[inline]
pub unsafe fn softmax_bwd_scalar_f64(
    grad: *const f64,
    output: *const f64,
    d_input: *mut f64,
    outer_size: usize,
    dim_size: usize,
) {
    for o in 0..outer_size {
        let base = o * dim_size;

        // Pass 1: dot = sum(grad * output)
        let mut dot = 0.0f64;
        for d in 0..dim_size {
            dot += *grad.add(base + d) * *output.add(base + d);
        }

        // Pass 2: d_input = output * (grad - dot)
        for d in 0..dim_size {
            let idx = base + d;
            *d_input.add(idx) = *output.add(idx) * (*grad.add(idx) - dot);
        }
    }
}

#[cfg(feature = "f16")]
/// f16 wrapper for softmax backward: processes one row at a time via f32 conversion.
///
/// # Safety
/// - All pointers must point to `outer_size * dim_size` elements
pub unsafe fn softmax_bwd_f16(
    grad: *const half::f16,
    output: *const half::f16,
    d_input: *mut half::f16,
    outer_size: usize,
    dim_size: usize,
) {
    use super::half_convert_utils::*;
    let row_len = dim_size;
    let mut grad_buf = vec![0.0f32; row_len];
    let mut out_buf = vec![0.0f32; row_len];
    let mut result_buf = vec![0.0f32; row_len];
    for i in 0..outer_size {
        let offset = i * dim_size;
        convert_f16_to_f32(
            grad.add(offset) as *const u16,
            grad_buf.as_mut_ptr(),
            row_len,
        );
        convert_f16_to_f32(
            output.add(offset) as *const u16,
            out_buf.as_mut_ptr(),
            row_len,
        );
        softmax_bwd_f32(
            grad_buf.as_ptr(),
            out_buf.as_ptr(),
            result_buf.as_mut_ptr(),
            1,
            dim_size,
        );
        convert_f32_to_f16(
            result_buf.as_ptr(),
            d_input.add(offset) as *mut u16,
            row_len,
        );
    }
}

#[cfg(feature = "f16")]
/// bf16 wrapper for softmax backward: processes one row at a time via f32 conversion.
///
/// # Safety
/// - All pointers must point to `outer_size * dim_size` elements
pub unsafe fn softmax_bwd_bf16(
    grad: *const half::bf16,
    output: *const half::bf16,
    d_input: *mut half::bf16,
    outer_size: usize,
    dim_size: usize,
) {
    use super::half_convert_utils::*;
    let row_len = dim_size;
    let mut grad_buf = vec![0.0f32; row_len];
    let mut out_buf = vec![0.0f32; row_len];
    let mut result_buf = vec![0.0f32; row_len];
    for i in 0..outer_size {
        let offset = i * dim_size;
        convert_bf16_to_f32(
            grad.add(offset) as *const u16,
            grad_buf.as_mut_ptr(),
            row_len,
        );
        convert_bf16_to_f32(
            output.add(offset) as *const u16,
            out_buf.as_mut_ptr(),
            row_len,
        );
        softmax_bwd_f32(
            grad_buf.as_ptr(),
            out_buf.as_ptr(),
            result_buf.as_mut_ptr(),
            1,
            dim_size,
        );
        convert_f32_to_bf16(
            result_buf.as_ptr(),
            d_input.add(offset) as *mut u16,
            row_len,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_bwd_f32() {
        // softmax output that sums to 1
        let output = [0.09003057f32, 0.24472847, 0.66524096]; // softmax([1,2,3])
        let grad = [1.0f32, 0.0, 0.0]; // d_loss/d_softmax
        let mut d_input = [0.0f32; 3];

        unsafe {
            softmax_bwd_f32(grad.as_ptr(), output.as_ptr(), d_input.as_mut_ptr(), 1, 3);
        }

        // dot = 1.0 * 0.09003057 = 0.09003057
        // d_input[0] = 0.09003057 * (1.0 - 0.09003057) = 0.0819
        // d_input[1] = 0.24472847 * (0.0 - 0.09003057) = -0.02203
        // d_input[2] = 0.66524096 * (0.0 - 0.09003057) = -0.05989
        assert!((d_input[0] - 0.08192507).abs() < 1e-5);
        assert!((d_input[1] - (-0.02203645)).abs() < 1e-5);
        assert!((d_input[2] - (-0.05988862)).abs() < 1e-5);

        // Gradients should sum to 0 (softmax outputs sum to 1, so Jacobian rows sum to 0)
        let sum: f32 = d_input.iter().sum();
        assert!(sum.abs() < 1e-6, "gradients should sum to 0, got {sum}");
    }

    #[test]
    fn test_softmax_bwd_simd() {
        let dim_size = 128;
        let outer_size = 4;

        // Create valid softmax outputs (sum to 1 per row)
        let mut output = vec![0.0f32; outer_size * dim_size];
        for o in 0..outer_size {
            let base = o * dim_size;
            let sum: f32 = (0..dim_size).map(|d| ((d as f32) * 0.1 - 5.0).exp()).sum();
            for d in 0..dim_size {
                output[base + d] = ((d as f32) * 0.1 - 5.0).exp() / sum;
            }
        }

        let grad: Vec<f32> = (0..(outer_size * dim_size))
            .map(|x| (x as f32) / 100.0 - 2.5)
            .collect();

        let mut d_input_simd = vec![0.0f32; outer_size * dim_size];
        let mut d_input_ref = vec![0.0f32; outer_size * dim_size];

        unsafe {
            softmax_bwd_f32(
                grad.as_ptr(),
                output.as_ptr(),
                d_input_simd.as_mut_ptr(),
                outer_size,
                dim_size,
            );
            softmax_bwd_scalar_f32(
                grad.as_ptr(),
                output.as_ptr(),
                d_input_ref.as_mut_ptr(),
                outer_size,
                dim_size,
            );
        }

        for i in 0..(outer_size * dim_size) {
            let rel_err = if d_input_ref[i].abs() > 1e-10 {
                (d_input_simd[i] - d_input_ref[i]).abs() / d_input_ref[i].abs()
            } else {
                (d_input_simd[i] - d_input_ref[i]).abs()
            };
            assert!(
                rel_err < 1e-3,
                "mismatch at {}: {} vs {} (rel_err: {})",
                i,
                d_input_simd[i],
                d_input_ref[i],
                rel_err
            );
        }
    }
}
