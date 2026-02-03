//! SIMD-optimized cumulative operations dispatch
//!
//! Provides AVX2/AVX-512 accelerated cumsum and cumprod strided kernels.
//! The strided kernels vectorize over the inner_size dimension, where each
//! SIMD lane maintains its own independent accumulator.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

use super::{SimdLevel, detect_simd};

/// Minimum inner_size to justify SIMD overhead for strided operations
const SIMD_THRESHOLD: usize = 16;

// ============================================================================
// Cumsum Strided - SIMD Dispatch
// ============================================================================

/// SIMD-optimized strided cumsum for f32.
///
/// Vectorizes over the inner_size dimension - each SIMD lane maintains
/// its own running sum independently.
///
/// # Safety
/// - All pointers must be valid for the specified sizes and strides
pub unsafe fn cumsum_strided_f32(
    a: *const f32,
    out: *mut f32,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    let level = detect_simd();

    if inner_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        cumsum_strided_scalar_f32(a, out, scan_size, outer_size, inner_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::cumsum_strided_f32(a, out, scan_size, outer_size, inner_size),
        SimdLevel::Avx2Fma => avx2::cumsum_strided_f32(a, out, scan_size, outer_size, inner_size),
        _ => cumsum_strided_scalar_f32(a, out, scan_size, outer_size, inner_size),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::cumsum_strided_f32(a, out, scan_size, outer_size, inner_size)
        }
        _ => cumsum_strided_scalar_f32(a, out, scan_size, outer_size, inner_size),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    cumsum_strided_scalar_f32(a, out, scan_size, outer_size, inner_size);
}

/// SIMD-optimized strided cumsum for f64.
pub unsafe fn cumsum_strided_f64(
    a: *const f64,
    out: *mut f64,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    let level = detect_simd();

    if inner_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        cumsum_strided_scalar_f64(a, out, scan_size, outer_size, inner_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::cumsum_strided_f64(a, out, scan_size, outer_size, inner_size),
        SimdLevel::Avx2Fma => avx2::cumsum_strided_f64(a, out, scan_size, outer_size, inner_size),
        _ => cumsum_strided_scalar_f64(a, out, scan_size, outer_size, inner_size),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::cumsum_strided_f64(a, out, scan_size, outer_size, inner_size)
        }
        _ => cumsum_strided_scalar_f64(a, out, scan_size, outer_size, inner_size),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    cumsum_strided_scalar_f64(a, out, scan_size, outer_size, inner_size);
}

// ============================================================================
// Cumprod Strided - SIMD Dispatch
// ============================================================================

/// SIMD-optimized strided cumprod for f32.
pub unsafe fn cumprod_strided_f32(
    a: *const f32,
    out: *mut f32,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    let level = detect_simd();

    if inner_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        cumprod_strided_scalar_f32(a, out, scan_size, outer_size, inner_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::cumprod_strided_f32(a, out, scan_size, outer_size, inner_size),
        SimdLevel::Avx2Fma => avx2::cumprod_strided_f32(a, out, scan_size, outer_size, inner_size),
        _ => cumprod_strided_scalar_f32(a, out, scan_size, outer_size, inner_size),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::cumprod_strided_f32(a, out, scan_size, outer_size, inner_size)
        }
        _ => cumprod_strided_scalar_f32(a, out, scan_size, outer_size, inner_size),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    cumprod_strided_scalar_f32(a, out, scan_size, outer_size, inner_size);
}

/// SIMD-optimized strided cumprod for f64.
pub unsafe fn cumprod_strided_f64(
    a: *const f64,
    out: *mut f64,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    let level = detect_simd();

    if inner_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        cumprod_strided_scalar_f64(a, out, scan_size, outer_size, inner_size);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::cumprod_strided_f64(a, out, scan_size, outer_size, inner_size),
        SimdLevel::Avx2Fma => avx2::cumprod_strided_f64(a, out, scan_size, outer_size, inner_size),
        _ => cumprod_strided_scalar_f64(a, out, scan_size, outer_size, inner_size),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::cumprod_strided_f64(a, out, scan_size, outer_size, inner_size)
        }
        _ => cumprod_strided_scalar_f64(a, out, scan_size, outer_size, inner_size),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    cumprod_strided_scalar_f64(a, out, scan_size, outer_size, inner_size);
}

// ============================================================================
// Scalar Fallbacks
// ============================================================================

#[inline]
unsafe fn cumsum_strided_scalar_f32(
    a: *const f32,
    out: *mut f32,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    for o in 0..outer_size {
        for i in 0..inner_size {
            let mut acc = 0.0f32;
            for s in 0..scan_size {
                let idx = o * scan_size * inner_size + s * inner_size + i;
                acc += *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

#[inline]
unsafe fn cumsum_strided_scalar_f64(
    a: *const f64,
    out: *mut f64,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    for o in 0..outer_size {
        for i in 0..inner_size {
            let mut acc = 0.0f64;
            for s in 0..scan_size {
                let idx = o * scan_size * inner_size + s * inner_size + i;
                acc += *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

#[inline]
unsafe fn cumprod_strided_scalar_f32(
    a: *const f32,
    out: *mut f32,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    for o in 0..outer_size {
        for i in 0..inner_size {
            let mut acc = 1.0f32;
            for s in 0..scan_size {
                let idx = o * scan_size * inner_size + s * inner_size + i;
                acc *= *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

#[inline]
unsafe fn cumprod_strided_scalar_f64(
    a: *const f64,
    out: *mut f64,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    for o in 0..outer_size {
        for i in 0..inner_size {
            let mut acc = 1.0f64;
            for s in 0..scan_size {
                let idx = o * scan_size * inner_size + s * inner_size + i;
                acc *= *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cumsum_strided_f32() {
        // 2 outer segments, scan_size=3, inner_size=4
        // Layout: [o=0,s=0,i=0-3], [o=0,s=1,i=0-3], [o=0,s=2,i=0-3],
        //         [o=1,s=0,i=0-3], [o=1,s=1,i=0-3], [o=1,s=2,i=0-3]
        let input: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let mut output = vec![0.0f32; 24];

        unsafe {
            cumsum_strided_f32(input.as_ptr(), output.as_mut_ptr(), 3, 2, 4);
        }

        // Check first outer segment (o=0), i=0
        // s=0: 0, s=1: 0+4=4, s=2: 0+4+8=12
        assert_eq!(output[0], 0.0);
        assert_eq!(output[4], 4.0);
        assert_eq!(output[8], 12.0);

        // Check first outer segment (o=0), i=1
        // s=0: 1, s=1: 1+5=6, s=2: 1+5+9=15
        assert_eq!(output[1], 1.0);
        assert_eq!(output[5], 6.0);
        assert_eq!(output[9], 15.0);
    }

    #[test]
    fn test_cumprod_strided_f32() {
        // 1 outer segment, scan_size=4, inner_size=2
        let input = vec![1.0f32, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 8];

        unsafe {
            cumprod_strided_f32(input.as_ptr(), output.as_mut_ptr(), 4, 1, 2);
        }

        // i=0: 1, 1*2=2, 2*3=6, 6*4=24
        assert_eq!(output[0], 1.0);
        assert_eq!(output[2], 2.0);
        assert_eq!(output[4], 6.0);
        assert_eq!(output[6], 24.0);

        // i=1: 2, 2*3=6, 6*4=24, 24*5=120
        assert_eq!(output[1], 2.0);
        assert_eq!(output[3], 6.0);
        assert_eq!(output[5], 24.0);
        assert_eq!(output[7], 120.0);
    }
}
