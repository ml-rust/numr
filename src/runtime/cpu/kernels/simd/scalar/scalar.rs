//! Scalar fallback implementations of tensor-scalar operations.

use crate::ops::BinaryOp;

/// Scalar fallback for f32
#[inline]
pub(super) unsafe fn scalar_scalar_f32(
    op: BinaryOp,
    a: *const f32,
    scalar: f32,
    out: *mut f32,
    len: usize,
) {
    match op {
        BinaryOp::Add => {
            for i in 0..len {
                *out.add(i) = *a.add(i) + scalar;
            }
        }
        BinaryOp::Sub => {
            for i in 0..len {
                *out.add(i) = *a.add(i) - scalar;
            }
        }
        BinaryOp::Mul => {
            for i in 0..len {
                *out.add(i) = *a.add(i) * scalar;
            }
        }
        BinaryOp::Div => {
            for i in 0..len {
                *out.add(i) = *a.add(i) / scalar;
            }
        }
        BinaryOp::Max => {
            for i in 0..len {
                let v = *a.add(i);
                *out.add(i) = if v > scalar { v } else { scalar };
            }
        }
        BinaryOp::Min => {
            for i in 0..len {
                let v = *a.add(i);
                *out.add(i) = if v < scalar { v } else { scalar };
            }
        }
        BinaryOp::Pow => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).powf(scalar);
            }
        }
        BinaryOp::Atan2 => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).atan2(scalar);
            }
        }
    }
}

/// Scalar fallback for f64
#[inline]
pub(super) unsafe fn scalar_scalar_f64(
    op: BinaryOp,
    a: *const f64,
    scalar: f64,
    out: *mut f64,
    len: usize,
) {
    match op {
        BinaryOp::Add => {
            for i in 0..len {
                *out.add(i) = *a.add(i) + scalar;
            }
        }
        BinaryOp::Sub => {
            for i in 0..len {
                *out.add(i) = *a.add(i) - scalar;
            }
        }
        BinaryOp::Mul => {
            for i in 0..len {
                *out.add(i) = *a.add(i) * scalar;
            }
        }
        BinaryOp::Div => {
            for i in 0..len {
                *out.add(i) = *a.add(i) / scalar;
            }
        }
        BinaryOp::Max => {
            for i in 0..len {
                let v = *a.add(i);
                *out.add(i) = if v > scalar { v } else { scalar };
            }
        }
        BinaryOp::Min => {
            for i in 0..len {
                let v = *a.add(i);
                *out.add(i) = if v < scalar { v } else { scalar };
            }
        }
        BinaryOp::Pow => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).powf(scalar);
            }
        }
        BinaryOp::Atan2 => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).atan2(scalar);
            }
        }
    }
}
