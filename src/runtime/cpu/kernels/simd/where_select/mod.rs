//! SIMD-accelerated conditional select (where) operation
//!
//! where(cond, x, y): out[i] = cond[i] ? x[i] : y[i]
//!
//! # SIMD Approach
//!
//! - Load condition bytes and expand to element-width masks
//! - Use SIMD blend operations based on mask

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

use super::{SimdLevel, detect_simd};

/// Minimum length to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

/// SIMD where for f32
///
/// # Safety
/// - `cond` must point to `len` u8 elements
/// - `x`, `y`, and `out` must point to `len` f32 elements
#[inline]
pub unsafe fn where_f32(cond: *const u8, x: *const f32, y: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        where_scalar_f32(cond, x, y, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::where_f32(cond, x, y, out, len),
        SimdLevel::Avx2Fma => avx2::where_f32(cond, x, y, out, len),
        _ => where_scalar_f32(cond, x, y, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::where_f32(cond, x, y, out, len),
        _ => where_scalar_f32(cond, x, y, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    where_scalar_f32(cond, x, y, out, len);
}

/// SIMD where for f64
///
/// # Safety
/// - `cond` must point to `len` u8 elements
/// - `x`, `y`, and `out` must point to `len` f64 elements
#[inline]
pub unsafe fn where_f64(cond: *const u8, x: *const f64, y: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        where_scalar_f64(cond, x, y, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::where_f64(cond, x, y, out, len),
        SimdLevel::Avx2Fma => avx2::where_f64(cond, x, y, out, len),
        _ => where_scalar_f64(cond, x, y, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::where_f64(cond, x, y, out, len),
        _ => where_scalar_f64(cond, x, y, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    where_scalar_f64(cond, x, y, out, len);
}

// ============================================================================
// Scalar fallbacks
// ============================================================================

/// Scalar where for f32
#[inline]
pub unsafe fn where_scalar_f32(
    cond: *const u8,
    x: *const f32,
    y: *const f32,
    out: *mut f32,
    len: usize,
) {
    for i in 0..len {
        *out.add(i) = if *cond.add(i) != 0 {
            *x.add(i)
        } else {
            *y.add(i)
        };
    }
}

/// Scalar where for f64
#[inline]
pub unsafe fn where_scalar_f64(
    cond: *const u8,
    x: *const f64,
    y: *const f64,
    out: *mut f64,
    len: usize,
) {
    for i in 0..len {
        *out.add(i) = if *cond.add(i) != 0 {
            *x.add(i)
        } else {
            *y.add(i)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_where_f32() {
        let len = 128;
        let cond: Vec<u8> = (0..len).map(|x| if x % 2 == 0 { 1 } else { 0 }).collect();
        let x: Vec<f32> = (0..len).map(|i| i as f32 * 10.0).collect();
        let y: Vec<f32> = (0..len).map(|i| -(i as f32)).collect();
        let mut out = vec![0.0f32; len];
        let mut out_ref = vec![0.0f32; len];

        unsafe {
            where_f32(cond.as_ptr(), x.as_ptr(), y.as_ptr(), out.as_mut_ptr(), len);
            where_scalar_f32(
                cond.as_ptr(),
                x.as_ptr(),
                y.as_ptr(),
                out_ref.as_mut_ptr(),
                len,
            );
        }

        assert_eq!(out, out_ref);
    }

    #[test]
    fn test_where_all_true() {
        let len = 64;
        let cond = vec![1u8; len];
        let x: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let y = vec![999.0f32; len];
        let mut out = vec![0.0f32; len];

        unsafe {
            where_f32(cond.as_ptr(), x.as_ptr(), y.as_ptr(), out.as_mut_ptr(), len);
        }

        assert_eq!(out, x);
    }

    #[test]
    fn test_where_all_false() {
        let len = 64;
        let cond = vec![0u8; len];
        let x = vec![999.0f32; len];
        let y: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; len];

        unsafe {
            where_f32(cond.as_ptr(), x.as_ptr(), y.as_ptr(), out.as_mut_ptr(), len);
        }

        assert_eq!(out, y);
    }
}
