//! SIMD-accelerated logsumexp operation
//!
//! Logsumexp: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
//! Commonly used in attention mechanisms and probability computations.
//!
//! # SIMD Optimizations
//!
//! - SIMD max-reduce for finding maximum
//! - SIMD exp computation (vectorized polynomial approximation)
//! - SIMD sum-reduce for accumulation

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

use super::{SimdLevel, detect_simd};

/// Minimum reduce size to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

/// SIMD logsumexp for f32
///
/// # Arguments
/// * `a` - Input pointer (reduce_size * outer_size elements)
/// * `out` - Output pointer (outer_size elements)
/// * `reduce_size` - Number of elements to reduce per segment
/// * `outer_size` - Number of independent reductions
///
/// # Safety
/// - `a` must point to `reduce_size * outer_size` elements
/// - `out` must point to `outer_size` elements
#[inline]
pub unsafe fn logsumexp_f32(a: *const f32, out: *mut f32, reduce_size: usize, outer_size: usize) {
    let level = detect_simd();

    if reduce_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        logsumexp_scalar_f32(a, out, reduce_size, outer_size);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::logsumexp_f32(a, out, reduce_size, outer_size),
        SimdLevel::Avx2Fma => avx2::logsumexp_f32(a, out, reduce_size, outer_size),
        SimdLevel::Scalar => unreachable!(),
    }
}

/// SIMD logsumexp for f64
///
/// # Safety
/// - `a` must point to `reduce_size * outer_size` elements
/// - `out` must point to `outer_size` elements
#[inline]
pub unsafe fn logsumexp_f64(a: *const f64, out: *mut f64, reduce_size: usize, outer_size: usize) {
    let level = detect_simd();

    if reduce_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        logsumexp_scalar_f64(a, out, reduce_size, outer_size);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::logsumexp_f64(a, out, reduce_size, outer_size),
        SimdLevel::Avx2Fma => avx2::logsumexp_f64(a, out, reduce_size, outer_size),
        SimdLevel::Scalar => unreachable!(),
    }
}

// ============================================================================
// Scalar fallbacks
// ============================================================================

/// Scalar logsumexp for f32
#[inline]
pub unsafe fn logsumexp_scalar_f32(
    a: *const f32,
    out: *mut f32,
    reduce_size: usize,
    outer_size: usize,
) {
    for o in 0..outer_size {
        let base = o * reduce_size;

        // Find max
        let mut max_val = *a.add(base);
        for i in 1..reduce_size {
            let val = *a.add(base + i);
            if val > max_val {
                max_val = val;
            }
        }

        // Compute sum(exp(x - max))
        let mut sum = 0.0f32;
        for i in 0..reduce_size {
            let val = *a.add(base + i);
            sum += (val - max_val).exp();
        }

        // Result = max + log(sum)
        *out.add(o) = max_val + sum.ln();
    }
}

/// Scalar logsumexp for f64
#[inline]
pub unsafe fn logsumexp_scalar_f64(
    a: *const f64,
    out: *mut f64,
    reduce_size: usize,
    outer_size: usize,
) {
    for o in 0..outer_size {
        let base = o * reduce_size;

        // Find max
        let mut max_val = *a.add(base);
        for i in 1..reduce_size {
            let val = *a.add(base + i);
            if val > max_val {
                max_val = val;
            }
        }

        // Compute sum(exp(x - max))
        let mut sum = 0.0f64;
        for i in 0..reduce_size {
            let val = *a.add(base + i);
            sum += (val - max_val).exp();
        }

        // Result = max + log(sum)
        *out.add(o) = max_val + sum.ln();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logsumexp_f32() {
        let reduce_size = 128;
        let outer_size = 4;
        let input: Vec<f32> = (0..(outer_size * reduce_size))
            .map(|x| (x as f32) / 100.0 - 2.5)
            .collect();
        let mut out = vec![0.0f32; outer_size];
        let mut out_ref = vec![0.0f32; outer_size];

        unsafe {
            logsumexp_f32(input.as_ptr(), out.as_mut_ptr(), reduce_size, outer_size);
            logsumexp_scalar_f32(
                input.as_ptr(),
                out_ref.as_mut_ptr(),
                reduce_size,
                outer_size,
            );
        }

        for i in 0..outer_size {
            let rel_err = if out_ref[i].abs() > 1e-10 {
                (out[i] - out_ref[i]).abs() / out_ref[i].abs()
            } else {
                (out[i] - out_ref[i]).abs()
            };
            assert!(
                rel_err < 1e-4,
                "mismatch at {}: {} vs {} (rel_err: {})",
                i,
                out[i],
                out_ref[i],
                rel_err
            );
        }
    }

    #[test]
    fn test_logsumexp_numerical_stability() {
        // Test with large values that would overflow naive exp
        let reduce_size = 64;
        let input: Vec<f32> = (0..reduce_size).map(|x| 1000.0 + x as f32).collect();
        let mut out = vec![0.0f32; 1];

        unsafe {
            logsumexp_f32(input.as_ptr(), out.as_mut_ptr(), reduce_size, 1);
        }

        // Should not be NaN or Inf
        assert!(out[0].is_finite(), "non-finite value: {}", out[0]);

        // Expected: max + log(sum(exp(x - max)))
        // = 1063 + log(exp(-63) + exp(-62) + ... + exp(0))
        // ≈ 1063 + log(1 + very small terms) ≈ 1063.46
        // Compute reference value
        let max_val = 1063.0f32;
        let sum: f32 = (0..reduce_size)
            .map(|x| ((1000.0 + x as f32) - max_val).exp())
            .sum();
        let expected = max_val + sum.ln();

        assert!(
            (out[0] - expected).abs() < 0.5,
            "result {} vs expected {}",
            out[0],
            expected
        );
    }

    #[test]
    fn test_logsumexp_single_element() {
        let input = [5.0f32];
        let mut out = [0.0f32];

        unsafe {
            logsumexp_scalar_f32(input.as_ptr(), out.as_mut_ptr(), 1, 1);
        }

        // logsumexp of single element = the element itself
        assert!((out[0] - 5.0).abs() < 1e-6);
    }
}
