//! f32/f64 specific scalar implementations for unary operations
//!
//! These are used by SIMD modules for tail handling and small array fallback.

use crate::ops::UnaryOp;

/// Scalar unary operation for f32 (used by SIMD for small arrays and tail)
#[inline]
pub unsafe fn unary_scalar_f32(op: UnaryOp, a: *const f32, out: *mut f32, len: usize) {
    match op {
        UnaryOp::Neg => {
            for i in 0..len {
                *out.add(i) = -(*a.add(i));
            }
        }
        UnaryOp::Abs => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).abs();
            }
        }
        UnaryOp::Sqrt => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).sqrt();
            }
        }
        UnaryOp::Square => {
            for i in 0..len {
                let v = *a.add(i);
                *out.add(i) = v * v;
            }
        }
        UnaryOp::Recip => {
            for i in 0..len {
                *out.add(i) = 1.0 / (*a.add(i));
            }
        }
        UnaryOp::Floor => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).floor();
            }
        }
        UnaryOp::Ceil => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).ceil();
            }
        }
        UnaryOp::Round => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).round();
            }
        }
        UnaryOp::Exp => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).exp();
            }
        }
        UnaryOp::Log => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).ln();
            }
        }
        UnaryOp::Sin => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).sin();
            }
        }
        UnaryOp::Cos => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).cos();
            }
        }
        UnaryOp::Tan => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).tan();
            }
        }
        UnaryOp::Tanh => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).tanh();
            }
        }
        UnaryOp::Sign => {
            for i in 0..len {
                let v = *a.add(i);
                *out.add(i) = if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                };
            }
        }
    }
}

/// Scalar unary operation for f64 (used by SIMD for small arrays and tail)
#[inline]
pub unsafe fn unary_scalar_f64(op: UnaryOp, a: *const f64, out: *mut f64, len: usize) {
    match op {
        UnaryOp::Neg => {
            for i in 0..len {
                *out.add(i) = -(*a.add(i));
            }
        }
        UnaryOp::Abs => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).abs();
            }
        }
        UnaryOp::Sqrt => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).sqrt();
            }
        }
        UnaryOp::Square => {
            for i in 0..len {
                let v = *a.add(i);
                *out.add(i) = v * v;
            }
        }
        UnaryOp::Recip => {
            for i in 0..len {
                *out.add(i) = 1.0 / (*a.add(i));
            }
        }
        UnaryOp::Floor => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).floor();
            }
        }
        UnaryOp::Ceil => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).ceil();
            }
        }
        UnaryOp::Round => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).round();
            }
        }
        UnaryOp::Exp => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).exp();
            }
        }
        UnaryOp::Log => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).ln();
            }
        }
        UnaryOp::Sin => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).sin();
            }
        }
        UnaryOp::Cos => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).cos();
            }
        }
        UnaryOp::Tan => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).tan();
            }
        }
        UnaryOp::Tanh => {
            for i in 0..len {
                *out.add(i) = (*a.add(i)).tanh();
            }
        }
        UnaryOp::Sign => {
            for i in 0..len {
                let v = *a.add(i);
                *out.add(i) = if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                };
            }
        }
    }
}

/// Scalar ReLU for f32 (used by SIMD for small arrays and tail)
#[inline]
pub unsafe fn relu_scalar_f32(a: *const f32, out: *mut f32, len: usize) {
    for i in 0..len {
        let v = *a.add(i);
        *out.add(i) = if v > 0.0 { v } else { 0.0 };
    }
}

/// Scalar ReLU for f64 (used by SIMD for small arrays and tail)
#[inline]
pub unsafe fn relu_scalar_f64(a: *const f64, out: *mut f64, len: usize) {
    for i in 0..len {
        let v = *a.add(i);
        *out.add(i) = if v > 0.0 { v } else { 0.0 };
    }
}
