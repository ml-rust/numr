//! SIMD-accelerated softmax operation
//!
//! Softmax is critical for attention mechanisms in transformers.
//! softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
//!
//! # SIMD Optimizations
//!
//! - SIMD max-reduce for finding maximum
//! - SIMD exp computation (vectorized)
//! - SIMD sum-reduce for normalization
//! - SIMD multiply for final division

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

use super::{SimdLevel, detect_simd};

/// Minimum dimension size to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

/// SIMD softmax for f32
///
/// # Arguments
/// * `a` - Input pointer (outer_size * dim_size elements)
/// * `out` - Output pointer (outer_size * dim_size elements)
/// * `outer_size` - Number of independent softmax operations
/// * `dim_size` - Size of the softmax dimension
///
/// # Safety
/// - `a` and `out` must point to `outer_size * dim_size` elements
#[inline]
pub unsafe fn softmax_f32(a: *const f32, out: *mut f32, outer_size: usize, dim_size: usize) {
    let level = detect_simd();

    if dim_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        softmax_scalar_f32(a, out, outer_size, dim_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::softmax_f32(a, out, outer_size, dim_size),
        SimdLevel::Avx2Fma => avx2::softmax_f32(a, out, outer_size, dim_size),
        _ => softmax_scalar_f32(a, out, outer_size, dim_size),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::softmax_f32(a, out, outer_size, dim_size)
        }
        _ => softmax_scalar_f32(a, out, outer_size, dim_size),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    softmax_scalar_f32(a, out, outer_size, dim_size);
}

/// SIMD softmax for f64
///
/// # Safety
/// - `a` and `out` must point to `outer_size * dim_size` elements
#[inline]
pub unsafe fn softmax_f64(a: *const f64, out: *mut f64, outer_size: usize, dim_size: usize) {
    let level = detect_simd();

    if dim_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        softmax_scalar_f64(a, out, outer_size, dim_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::softmax_f64(a, out, outer_size, dim_size),
        SimdLevel::Avx2Fma => avx2::softmax_f64(a, out, outer_size, dim_size),
        _ => softmax_scalar_f64(a, out, outer_size, dim_size),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::softmax_f64(a, out, outer_size, dim_size)
        }
        _ => softmax_scalar_f64(a, out, outer_size, dim_size),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    softmax_scalar_f64(a, out, outer_size, dim_size);
}

// ============================================================================
// Scalar fallbacks
// ============================================================================

/// Scalar softmax for f32
#[inline]
pub unsafe fn softmax_scalar_f32(a: *const f32, out: *mut f32, outer_size: usize, dim_size: usize) {
    for o in 0..outer_size {
        let base = o * dim_size;

        // Find max
        let mut max_val = *a.add(base);
        for d in 1..dim_size {
            let val = *a.add(base + d);
            if val > max_val {
                max_val = val;
            }
        }

        // Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        for d in 0..dim_size {
            let val = *a.add(base + d);
            let exp_val = (val - max_val).exp();
            *out.add(base + d) = exp_val;
            sum += exp_val;
        }

        // Normalize
        let inv_sum = 1.0 / sum;
        for d in 0..dim_size {
            *out.add(base + d) *= inv_sum;
        }
    }
}

/// Scalar softmax for f64
#[inline]
pub unsafe fn softmax_scalar_f64(a: *const f64, out: *mut f64, outer_size: usize, dim_size: usize) {
    for o in 0..outer_size {
        let base = o * dim_size;

        // Find max
        let mut max_val = *a.add(base);
        for d in 1..dim_size {
            let val = *a.add(base + d);
            if val > max_val {
                max_val = val;
            }
        }

        // Compute exp(x - max) and sum
        let mut sum = 0.0f64;
        for d in 0..dim_size {
            let val = *a.add(base + d);
            let exp_val = (val - max_val).exp();
            *out.add(base + d) = exp_val;
            sum += exp_val;
        }

        // Normalize
        let inv_sum = 1.0 / sum;
        for d in 0..dim_size {
            *out.add(base + d) *= inv_sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_f32() {
        let dim_size = 128;
        let outer_size = 4;
        let input: Vec<f32> = (0..(outer_size * dim_size))
            .map(|x| (x as f32) / 100.0 - 2.5)
            .collect();
        let mut out = vec![0.0f32; outer_size * dim_size];
        let mut out_ref = vec![0.0f32; outer_size * dim_size];

        unsafe {
            softmax_f32(input.as_ptr(), out.as_mut_ptr(), outer_size, dim_size);
            softmax_scalar_f32(input.as_ptr(), out_ref.as_mut_ptr(), outer_size, dim_size);
        }

        for i in 0..(outer_size * dim_size) {
            // SIMD exp approximation has ~1e-4 relative error
            let rel_err = if out_ref[i].abs() > 1e-10 {
                (out[i] - out_ref[i]).abs() / out_ref[i].abs()
            } else {
                (out[i] - out_ref[i]).abs()
            };
            assert!(
                rel_err < 1e-3,
                "mismatch at {}: {} vs {} (rel_err: {})",
                i,
                out[i],
                out_ref[i],
                rel_err
            );
        }
    }

    #[test]
    fn test_softmax_sum_to_one() {
        let dim_size = 64;
        let outer_size = 2;
        let input: Vec<f32> = (0..(outer_size * dim_size))
            .map(|x| (x as f32) * 0.1 - 3.0)
            .collect();
        let mut out = vec![0.0f32; outer_size * dim_size];

        unsafe {
            softmax_f32(input.as_ptr(), out.as_mut_ptr(), outer_size, dim_size);
        }

        // Each row should sum to 1.0
        for o in 0..outer_size {
            let row_sum: f32 = out[o * dim_size..(o + 1) * dim_size].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "row {} sum = {}, expected 1.0",
                o,
                row_sum
            );
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Test with large values that would overflow naive exp
        let dim_size = 64;
        let input: Vec<f32> = (0..dim_size).map(|x| 1000.0 + x as f32).collect();
        let mut out = vec![0.0f32; dim_size];

        unsafe {
            softmax_f32(input.as_ptr(), out.as_mut_ptr(), 1, dim_size);
        }

        // Should not be NaN or Inf
        for (i, &val) in out.iter().enumerate() {
            assert!(val.is_finite(), "non-finite value at {}: {}", i, val);
        }

        // Should sum to 1.0
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {}, expected 1.0", sum);
    }
}
