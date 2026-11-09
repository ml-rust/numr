//! Original-value classification for exceptional SIMD max reductions.

use crate::dtype::Element;

/// Scan only exceptional maxima so SIMD max NaN suppression cannot hide masked NaNs.
#[inline]
pub(super) unsafe fn original_row_limit<T: Element>(base: *const T, len: usize) -> Option<T> {
    let mut all_negative_infinity = true;
    let mut any_nan = false;
    for i in 0..len {
        let value = (*base.add(i)).to_f64();
        all_negative_infinity &= value == f64::NEG_INFINITY;
        any_nan |= value.is_nan();
    }
    if all_negative_infinity {
        Some(T::from_f64(f64::NEG_INFINITY))
    } else if any_nan {
        Some(T::from_f64(f64::NAN))
    } else {
        None
    }
}
