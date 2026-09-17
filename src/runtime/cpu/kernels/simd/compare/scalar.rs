//! Scalar fallbacks for SIMD comparison operations

use crate::ops::CompareOp;

/// Scalar comparison for f32
#[inline]
pub unsafe fn compare_scalar_f32(
    op: CompareOp,
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    len: usize,
) {
    match op {
        CompareOp::Eq => {
            for i in 0..len {
                *out.add(i) = if *a.add(i) == *b.add(i) { 1.0 } else { 0.0 };
            }
        }
        CompareOp::Ne => {
            for i in 0..len {
                *out.add(i) = if *a.add(i) != *b.add(i) { 1.0 } else { 0.0 };
            }
        }
        CompareOp::Lt => {
            for i in 0..len {
                *out.add(i) = if *a.add(i) < *b.add(i) { 1.0 } else { 0.0 };
            }
        }
        CompareOp::Le => {
            for i in 0..len {
                *out.add(i) = if *a.add(i) <= *b.add(i) { 1.0 } else { 0.0 };
            }
        }
        CompareOp::Gt => {
            for i in 0..len {
                *out.add(i) = if *a.add(i) > *b.add(i) { 1.0 } else { 0.0 };
            }
        }
        CompareOp::Ge => {
            for i in 0..len {
                *out.add(i) = if *a.add(i) >= *b.add(i) { 1.0 } else { 0.0 };
            }
        }
    }
}

/// Scalar comparison for f64
#[inline]
pub unsafe fn compare_scalar_f64(
    op: CompareOp,
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    len: usize,
) {
    match op {
        CompareOp::Eq => {
            for i in 0..len {
                *out.add(i) = if *a.add(i) == *b.add(i) { 1.0 } else { 0.0 };
            }
        }
        CompareOp::Ne => {
            for i in 0..len {
                *out.add(i) = if *a.add(i) != *b.add(i) { 1.0 } else { 0.0 };
            }
        }
        CompareOp::Lt => {
            for i in 0..len {
                *out.add(i) = if *a.add(i) < *b.add(i) { 1.0 } else { 0.0 };
            }
        }
        CompareOp::Le => {
            for i in 0..len {
                *out.add(i) = if *a.add(i) <= *b.add(i) { 1.0 } else { 0.0 };
            }
        }
        CompareOp::Gt => {
            for i in 0..len {
                *out.add(i) = if *a.add(i) > *b.add(i) { 1.0 } else { 0.0 };
            }
        }
        CompareOp::Ge => {
            for i in 0..len {
                *out.add(i) = if *a.add(i) >= *b.add(i) { 1.0 } else { 0.0 };
            }
        }
    }
}
