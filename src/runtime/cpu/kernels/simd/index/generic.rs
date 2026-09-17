//! Generic Element-typed wrappers over the f32/f64 SIMD dispatch.

use super::dispatch::{masked_fill_f32, masked_fill_f64, masked_select_f32, masked_select_f64};
use crate::dtype::Element;

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
