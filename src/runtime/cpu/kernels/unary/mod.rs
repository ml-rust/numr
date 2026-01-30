//! Unary and activation operation kernels
//!
//! Provides element-wise unary operations with automatic SIMD dispatch.
//! On x86-64, f32 and f64 operations use AVX-512 or AVX2 when available.

pub mod scalar;

pub use scalar::{relu_scalar_f32, relu_scalar_f64, unary_scalar_f32, unary_scalar_f64};

use crate::dtype::{DType, Element};
use crate::ops::UnaryOp;

/// Execute a unary operation element-wise with automatic SIMD dispatch
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration
/// - AVX2: 8 f32s or 4 f64s per iteration
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn unary_op_kernel<T: Element>(op: UnaryOp, a: *const T, out: *mut T, len: usize) {
    // Dispatch to SIMD for f32/f64 on x86-64
    #[cfg(target_arch = "x86_64")]
    {
        use super::simd::unary;

        match T::DTYPE {
            DType::F32 => {
                unary::unary_f32(op, a as *const f32, out as *mut f32, len);
                return;
            }
            DType::F64 => {
                unary::unary_f64(op, a as *const f64, out as *mut f64, len);
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback
    unary_op_scalar(op, a, out, len);
}

/// Scalar unary operation for all Element types
#[inline]
unsafe fn unary_op_scalar<T: Element>(op: UnaryOp, a: *const T, out: *mut T, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    match op {
        UnaryOp::Neg => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(-v);
            }
        }
        UnaryOp::Abs => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.abs());
            }
        }
        UnaryOp::Sqrt => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.sqrt());
            }
        }
        UnaryOp::Exp => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.exp());
            }
        }
        UnaryOp::Log => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.ln());
            }
        }
        UnaryOp::Sin => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.sin());
            }
        }
        UnaryOp::Cos => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.cos());
            }
        }
        UnaryOp::Tan => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.tan());
            }
        }
        UnaryOp::Tanh => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.tanh());
            }
        }
        UnaryOp::Recip => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(1.0 / v);
            }
        }
        UnaryOp::Square => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v * v);
            }
        }
        UnaryOp::Floor => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.floor());
            }
        }
        UnaryOp::Ceil => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.ceil());
            }
        }
        UnaryOp::Round => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                out_slice[i] = T::from_f64(v.round());
            }
        }
        UnaryOp::Sign => {
            for i in 0..len {
                let v = a_slice[i].to_f64();
                let sign = if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                out_slice[i] = T::from_f64(sign);
            }
        }
    }
}

/// ReLU activation: max(0, x) with automatic SIMD dispatch
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn relu_kernel<T: Element>(a: *const T, out: *mut T, len: usize) {
    // Dispatch to SIMD for f32/f64 on x86-64
    #[cfg(target_arch = "x86_64")]
    {
        use super::simd::unary;

        match T::DTYPE {
            DType::F32 => {
                unary::relu_f32(a as *const f32, out as *mut f32, len);
                return;
            }
            DType::F64 => {
                unary::relu_f64(a as *const f64, out as *mut f64, len);
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback
    relu_scalar(a, out, len);
}

/// Scalar ReLU for all Element types
#[inline]
unsafe fn relu_scalar<T: Element>(a: *const T, out: *mut T, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);
    let zero = T::zero();

    for i in 0..len {
        out_slice[i] = if a_slice[i] > zero { a_slice[i] } else { zero };
    }
}

/// Sigmoid activation: 1 / (1 + exp(-x))
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn sigmoid_kernel<T: Element>(a: *const T, out: *mut T, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let v = a_slice[i].to_f64();
        let sig = 1.0 / (1.0 + (-v).exp());
        out_slice[i] = T::from_f64(sig);
    }
}

/// SiLU (Swish) activation: x / (1 + exp(-x)) = x * sigmoid(x)
///
/// Used in LLaMA, Mistral, and other modern transformer architectures.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn silu_kernel<T: Element>(a: *const T, out: *mut T, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let x = a_slice[i].to_f64();
        // SiLU(x) = x / (1 + exp(-x)) = x * sigmoid(x)
        let result = x / (1.0 + (-x).exp());
        out_slice[i] = T::from_f64(result);
    }
}

/// GELU (Gaussian Error Linear Unit) activation using tanh approximation
///
/// Computes: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// Used in GPT, BERT, and other transformer architectures.
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn gelu_kernel<T: Element>(a: *const T, out: *mut T, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    // GELU constants
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654; // sqrt(2/pi)
    const TANH_COEF: f64 = 0.044715;

    for i in 0..len {
        let x = a_slice[i].to_f64();
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let inner = SQRT_2_OVER_PI * (x + TANH_COEF * x * x * x);
        let result = 0.5 * x * (1.0 + inner.tanh());
        out_slice[i] = T::from_f64(result);
    }
}

/// Leaky ReLU activation: max(negative_slope * x, x)
///
/// # Safety
/// - `a` must be valid pointer to `len` elements
/// - `out` must be valid pointer to `len` elements (may alias `a`)
pub unsafe fn leaky_relu_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    len: usize,
    negative_slope: f64,
) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);
    let zero = T::zero();

    for i in 0..len {
        let x = a_slice[i];
        out_slice[i] = if x > zero {
            x
        } else {
            T::from_f64(x.to_f64() * negative_slope)
        };
    }
}

/// ELU activation: x if x > 0, else alpha * (exp(x) - 1)
///
/// # Safety
/// - `a` must be valid pointer to `len` elements
/// - `out` must be valid pointer to `len` elements (may alias `a`)
pub unsafe fn elu_kernel<T: Element>(a: *const T, out: *mut T, len: usize, alpha: f64) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);
    let zero = T::zero();

    for i in 0..len {
        let x = a_slice[i];
        out_slice[i] = if x > zero {
            x
        } else {
            T::from_f64(alpha * (x.to_f64().exp() - 1.0))
        };
    }
}

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

/// Clamp values to a range: out[i] = min(max(a[i], min_val), max_val)
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn clamp_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    len: usize,
    min_val: f64,
    max_val: f64,
) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let val = a_slice[i].to_f64();
        let clamped = if val < min_val {
            min_val
        } else if val > max_val {
            max_val
        } else {
            val
        };
        out_slice[i] = T::from_f64(clamped);
    }
}
