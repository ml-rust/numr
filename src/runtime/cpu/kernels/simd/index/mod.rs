//! SIMD-optimized index operations dispatch
//!
//! Provides AVX2/AVX-512 accelerated masked_fill and masked_select operations.

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

use super::{SimdLevel, detect_simd};
use crate::dtype::Element;

/// Minimum elements to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

// ============================================================================
// Masked Fill - SIMD Dispatch
// ============================================================================

/// SIMD-optimized masked fill for f32.
///
/// Fills output with `value` where mask is true, otherwise copies from input.
///
/// # Safety
/// - All pointers must be valid for `len` elements
pub unsafe fn masked_fill_f32(
    input: *const f32,
    mask: *const u8,
    output: *mut f32,
    len: usize,
    value: f32,
) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        masked_fill_scalar_f32(input, mask, output, len, value);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::masked_fill_f32(input, mask, output, len, value),
        SimdLevel::Avx2Fma => avx2::masked_fill_f32(input, mask, output, len, value),
        _ => masked_fill_scalar_f32(input, mask, output, len, value),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::masked_fill_f32(input, mask, output, len, value)
        }
        _ => masked_fill_scalar_f32(input, mask, output, len, value),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    masked_fill_scalar_f32(input, mask, output, len, value);
}

/// SIMD-optimized masked fill for f64.
pub unsafe fn masked_fill_f64(
    input: *const f64,
    mask: *const u8,
    output: *mut f64,
    len: usize,
    value: f64,
) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        masked_fill_scalar_f64(input, mask, output, len, value);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::masked_fill_f64(input, mask, output, len, value),
        SimdLevel::Avx2Fma => avx2::masked_fill_f64(input, mask, output, len, value),
        _ => masked_fill_scalar_f64(input, mask, output, len, value),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::masked_fill_f64(input, mask, output, len, value)
        }
        _ => masked_fill_scalar_f64(input, mask, output, len, value),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    masked_fill_scalar_f64(input, mask, output, len, value);
}

// ============================================================================
// Masked Select - SIMD Dispatch
// ============================================================================

/// SIMD-optimized masked select for f32.
///
/// Selects elements where mask is true into a contiguous output.
///
/// # Safety
/// - All pointers must be valid
/// - `output` must have space for at least `len` elements (worst case all selected)
pub unsafe fn masked_select_f32(
    input: *const f32,
    mask: *const u8,
    output: *mut f32,
    len: usize,
) -> usize {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        return masked_select_scalar_f32(input, mask, output, len);
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => return avx512::masked_select_f32(input, mask, output, len),
        SimdLevel::Avx2Fma => return avx2::masked_select_f32(input, mask, output, len),
        _ => return masked_select_scalar_f32(input, mask, output, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            return aarch64::neon::masked_select_f32(input, mask, output, len);
        }
        _ => return masked_select_scalar_f32(input, mask, output, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    masked_select_scalar_f32(input, mask, output, len)
}

/// SIMD-optimized masked select for f64.
pub unsafe fn masked_select_f64(
    input: *const f64,
    mask: *const u8,
    output: *mut f64,
    len: usize,
) -> usize {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        return masked_select_scalar_f64(input, mask, output, len);
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => return avx512::masked_select_f64(input, mask, output, len),
        SimdLevel::Avx2Fma => return avx2::masked_select_f64(input, mask, output, len),
        _ => return masked_select_scalar_f64(input, mask, output, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            return aarch64::neon::masked_select_f64(input, mask, output, len);
        }
        _ => return masked_select_scalar_f64(input, mask, output, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    masked_select_scalar_f64(input, mask, output, len)
}

// ============================================================================
// Masked Count - SIMD Dispatch
// ============================================================================

/// SIMD-optimized mask count (popcount).
///
/// Counts the number of true elements in the mask.
pub unsafe fn masked_count(mask: *const u8, len: usize) -> usize {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        return masked_count_scalar(mask, len);
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => return avx512::masked_count(mask, len),
        SimdLevel::Avx2Fma => return avx2::masked_count(mask, len),
        _ => return masked_count_scalar(mask, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => return aarch64::neon::masked_count(mask, len),
        _ => return masked_count_scalar(mask, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    masked_count_scalar(mask, len)
}

// ============================================================================
// Scalar Fallbacks
// ============================================================================

#[inline]
unsafe fn masked_fill_scalar_f32(
    input: *const f32,
    mask: *const u8,
    output: *mut f32,
    len: usize,
    value: f32,
) {
    for i in 0..len {
        *output.add(i) = if *mask.add(i) != 0 {
            value
        } else {
            *input.add(i)
        };
    }
}

#[inline]
unsafe fn masked_fill_scalar_f64(
    input: *const f64,
    mask: *const u8,
    output: *mut f64,
    len: usize,
    value: f64,
) {
    for i in 0..len {
        *output.add(i) = if *mask.add(i) != 0 {
            value
        } else {
            *input.add(i)
        };
    }
}

#[inline]
unsafe fn masked_select_scalar_f32(
    input: *const f32,
    mask: *const u8,
    output: *mut f32,
    len: usize,
) -> usize {
    let mut out_idx = 0;
    for i in 0..len {
        if *mask.add(i) != 0 {
            *output.add(out_idx) = *input.add(i);
            out_idx += 1;
        }
    }
    out_idx
}

#[inline]
unsafe fn masked_select_scalar_f64(
    input: *const f64,
    mask: *const u8,
    output: *mut f64,
    len: usize,
) -> usize {
    let mut out_idx = 0;
    for i in 0..len {
        if *mask.add(i) != 0 {
            *output.add(out_idx) = *input.add(i);
            out_idx += 1;
        }
    }
    out_idx
}

#[inline]
unsafe fn masked_count_scalar(mask: *const u8, len: usize) -> usize {
    let mut count = 0;
    for i in 0..len {
        if *mask.add(i) != 0 {
            count += 1;
        }
    }
    count
}

// ============================================================================
// Generic Dispatcher
// ============================================================================

/// Generic masked fill for any Element type.
///
/// Uses SIMD for f32/f64, scalar for other types.
#[allow(dead_code)]
pub unsafe fn masked_fill<T: Element>(
    input: *const T,
    mask: *const u8,
    output: *mut T,
    len: usize,
    value: f64,
) {
    // For f32/f64, use SIMD paths
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        masked_fill_f32(
            input as *const f32,
            mask,
            output as *mut f32,
            len,
            value as f32,
        );
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        masked_fill_f64(input as *const f64, mask, output as *mut f64, len, value);
    } else {
        // Scalar fallback for other types
        let fill_val = T::from_f64(value);
        for i in 0..len {
            *output.add(i) = if *mask.add(i) != 0 {
                fill_val
            } else {
                *input.add(i)
            };
        }
    }
}

/// Generic masked select for any Element type.
#[allow(dead_code)]
pub unsafe fn masked_select<T: Element>(
    input: *const T,
    mask: *const u8,
    output: *mut T,
    len: usize,
) -> usize {
    // For f32/f64, use SIMD paths
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        masked_select_f32(input as *const f32, mask, output as *mut f32, len)
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        masked_select_f64(input as *const f64, mask, output as *mut f64, len)
    } else {
        // Scalar fallback for other types
        let mut out_idx = 0;
        for i in 0..len {
            if *mask.add(i) != 0 {
                *output.add(out_idx) = *input.add(i);
                out_idx += 1;
            }
        }
        out_idx
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masked_fill_f32() {
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let mask: Vec<u8> = (0..100).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
        let mut output = vec![0.0f32; 100];
        let fill_value = -999.0f32;

        unsafe {
            masked_fill_f32(
                input.as_ptr(),
                mask.as_ptr(),
                output.as_mut_ptr(),
                100,
                fill_value,
            );
        }

        for i in 0..100 {
            let expected = if i % 3 == 0 { fill_value } else { i as f32 };
            assert_eq!(output[i], expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_masked_select_f32() {
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let mask: Vec<u8> = (0..100).map(|i| if i % 5 == 0 { 1 } else { 0 }).collect();
        let mut output = vec![0.0f32; 100];

        let count =
            unsafe { masked_select_f32(input.as_ptr(), mask.as_ptr(), output.as_mut_ptr(), 100) };

        assert_eq!(count, 20); // 0, 5, 10, 15, ..., 95 = 20 elements

        for (j, i) in (0..100).filter(|i| i % 5 == 0).enumerate() {
            assert_eq!(output[j], i as f32, "mismatch at output index {}", j);
        }
    }

    #[test]
    fn test_masked_count() {
        let mask: Vec<u8> = (0..256).map(|i| if i % 4 == 0 { 1 } else { 0 }).collect();

        let count = unsafe { masked_count(mask.as_ptr(), 256) };

        assert_eq!(count, 64); // 256 / 4 = 64
    }
}
