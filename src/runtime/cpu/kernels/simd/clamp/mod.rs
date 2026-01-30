//! SIMD-accelerated clamp operation
//!
//! clamp(x, min, max) = min(max(x, min), max)
//!
//! # SIMD Approach
//!
//! - Broadcast min and max values to vectors
//! - Use SIMD max then min operations

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

use super::{SimdLevel, detect_simd};

/// Minimum length to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

/// SIMD clamp for f32
///
/// # Safety
/// - `a` and `out` must point to `len` elements
#[inline]
pub unsafe fn clamp_f32(a: *const f32, out: *mut f32, len: usize, min_val: f32, max_val: f32) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        clamp_scalar_f32(a, out, len, min_val, max_val);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::clamp_f32(a, out, len, min_val, max_val),
        SimdLevel::Avx2Fma => avx2::clamp_f32(a, out, len, min_val, max_val),
        SimdLevel::Scalar => unreachable!(),
    }
}

/// SIMD clamp for f64
///
/// # Safety
/// - `a` and `out` must point to `len` elements
#[inline]
pub unsafe fn clamp_f64(a: *const f64, out: *mut f64, len: usize, min_val: f64, max_val: f64) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        clamp_scalar_f64(a, out, len, min_val, max_val);
        return;
    }

    match level {
        SimdLevel::Avx512 => avx512::clamp_f64(a, out, len, min_val, max_val),
        SimdLevel::Avx2Fma => avx2::clamp_f64(a, out, len, min_val, max_val),
        SimdLevel::Scalar => unreachable!(),
    }
}

// ============================================================================
// Scalar fallbacks
// ============================================================================

/// Scalar clamp for f32
#[inline]
pub unsafe fn clamp_scalar_f32(
    a: *const f32,
    out: *mut f32,
    len: usize,
    min_val: f32,
    max_val: f32,
) {
    for i in 0..len {
        let val = *a.add(i);
        *out.add(i) = val.max(min_val).min(max_val);
    }
}

/// Scalar clamp for f64
#[inline]
pub unsafe fn clamp_scalar_f64(
    a: *const f64,
    out: *mut f64,
    len: usize,
    min_val: f64,
    max_val: f64,
) {
    for i in 0..len {
        let val = *a.add(i);
        *out.add(i) = val.max(min_val).min(max_val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_f32() {
        let len = 128;
        let input: Vec<f32> = (0..len).map(|x| (x as f32) - 64.0).collect();
        let mut out = vec![0.0f32; len];
        let mut out_ref = vec![0.0f32; len];

        let min_val = -10.0f32;
        let max_val = 10.0f32;

        unsafe {
            clamp_f32(input.as_ptr(), out.as_mut_ptr(), len, min_val, max_val);
            clamp_scalar_f32(input.as_ptr(), out_ref.as_mut_ptr(), len, min_val, max_val);
        }

        assert_eq!(out, out_ref);
    }

    #[test]
    fn test_clamp_all_below() {
        let len = 64;
        let input = vec![-100.0f32; len];
        let mut out = vec![0.0f32; len];

        unsafe {
            clamp_f32(input.as_ptr(), out.as_mut_ptr(), len, 0.0, 10.0);
        }

        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_clamp_all_above() {
        let len = 64;
        let input = vec![100.0f32; len];
        let mut out = vec![0.0f32; len];

        unsafe {
            clamp_f32(input.as_ptr(), out.as_mut_ptr(), len, 0.0, 10.0);
        }

        assert!(out.iter().all(|&x| x == 10.0));
    }

    #[test]
    fn test_clamp_in_range() {
        let len = 64;
        let input: Vec<f32> = (0..len).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; len];

        unsafe {
            clamp_f32(input.as_ptr(), out.as_mut_ptr(), len, -100.0, 100.0);
        }

        assert_eq!(out, input);
    }
}
