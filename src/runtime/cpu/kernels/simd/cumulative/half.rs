//! f16 / bf16 wrappers for strided cumulative operations.
//!
//! Converts to f32, runs the f32 SIMD kernel, converts the result back.

#[cfg(feature = "f16")]
use super::dispatch::{cumprod_strided_f32, cumsum_strided_f32};

#[cfg(feature = "f16")]
/// f16 wrapper for cumsum_strided: converts input to f32, runs f32 cumsum, converts output back.
///
/// # Safety
/// - All pointers must be valid for the specified sizes
pub unsafe fn cumsum_strided_f16(
    a: *const half::f16,
    out: *mut half::f16,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    use super::super::half_convert_utils::*;
    let total = outer_size * scan_size * inner_size;
    let mut a_f32 = vec![0.0f32; total];
    let mut out_f32 = vec![0.0f32; total];
    convert_f16_to_f32(a as *const u16, a_f32.as_mut_ptr(), total);
    cumsum_strided_f32(
        a_f32.as_ptr(),
        out_f32.as_mut_ptr(),
        scan_size,
        outer_size,
        inner_size,
    );
    convert_f32_to_f16(out_f32.as_ptr(), out as *mut u16, total);
}

#[cfg(feature = "f16")]
/// bf16 wrapper for cumsum_strided: converts input to f32, runs f32 cumsum, converts output back.
///
/// # Safety
/// - All pointers must be valid for the specified sizes
pub unsafe fn cumsum_strided_bf16(
    a: *const half::bf16,
    out: *mut half::bf16,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    use super::super::half_convert_utils::*;
    let total = outer_size * scan_size * inner_size;
    let mut a_f32 = vec![0.0f32; total];
    let mut out_f32 = vec![0.0f32; total];
    convert_bf16_to_f32(a as *const u16, a_f32.as_mut_ptr(), total);
    cumsum_strided_f32(
        a_f32.as_ptr(),
        out_f32.as_mut_ptr(),
        scan_size,
        outer_size,
        inner_size,
    );
    convert_f32_to_bf16(out_f32.as_ptr(), out as *mut u16, total);
}

#[cfg(feature = "f16")]
/// f16 wrapper for cumprod_strided: converts input to f32, runs f32 cumprod, converts output back.
///
/// # Safety
/// - All pointers must be valid for the specified sizes
pub unsafe fn cumprod_strided_f16(
    a: *const half::f16,
    out: *mut half::f16,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    use super::super::half_convert_utils::*;
    let total = outer_size * scan_size * inner_size;
    let mut a_f32 = vec![0.0f32; total];
    let mut out_f32 = vec![0.0f32; total];
    convert_f16_to_f32(a as *const u16, a_f32.as_mut_ptr(), total);
    cumprod_strided_f32(
        a_f32.as_ptr(),
        out_f32.as_mut_ptr(),
        scan_size,
        outer_size,
        inner_size,
    );
    convert_f32_to_f16(out_f32.as_ptr(), out as *mut u16, total);
}

#[cfg(feature = "f16")]
/// bf16 wrapper for cumprod_strided: converts input to f32, runs f32 cumprod, converts output back.
///
/// # Safety
/// - All pointers must be valid for the specified sizes
pub unsafe fn cumprod_strided_bf16(
    a: *const half::bf16,
    out: *mut half::bf16,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    use super::super::half_convert_utils::*;
    let total = outer_size * scan_size * inner_size;
    let mut a_f32 = vec![0.0f32; total];
    let mut out_f32 = vec![0.0f32; total];
    convert_bf16_to_f32(a as *const u16, a_f32.as_mut_ptr(), total);
    cumprod_strided_f32(
        a_f32.as_ptr(),
        out_f32.as_mut_ptr(),
        scan_size,
        outer_size,
        inner_size,
    );
    convert_f32_to_bf16(out_f32.as_ptr(), out as *mut u16, total);
}
