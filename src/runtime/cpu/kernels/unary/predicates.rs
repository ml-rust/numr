//! Element-wise NaN/Inf predicate kernels.

use crate::dtype::Element;

/// Check for NaN values element-wise
///
/// Returns 1 (u8) if the value is NaN, 0 otherwise.
///
/// # Safety
/// - `a` must be valid pointer to `len` elements
/// - `out` must be valid pointer to `len` u8 elements
#[inline]
pub unsafe fn isnan_kernel<T: Element>(a: *const T, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let v = a_slice[i].to_f64();
        out_slice[i] = if v.is_nan() { 1 } else { 0 };
    }
}

/// Check for Inf values element-wise
///
/// Returns 1 (u8) if the value is infinite (positive or negative), 0 otherwise.
///
/// # Safety
/// - `a` must be valid pointer to `len` elements
/// - `out` must be valid pointer to `len` u8 elements
#[inline]
pub unsafe fn isinf_kernel<T: Element>(a: *const T, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let v = a_slice[i].to_f64();
        out_slice[i] = if v.is_infinite() { 1 } else { 0 };
    }
}
