//! Scalar fallbacks for conditional select (where) operation

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
