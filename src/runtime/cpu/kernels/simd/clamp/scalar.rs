//! Scalar fallbacks for clamp operation

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
