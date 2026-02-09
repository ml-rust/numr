//! Neural network activation function kernels
//!
//! Provides element-wise activation functions with automatic SIMD dispatch.
//! On x86-64, f32 and f64 operations use AVX-512 or AVX2 when available.

use crate::dtype::{DType, Element};

/// Sigmoid activation: 1 / (1 + exp(-x))
///
/// On x86-64, dispatches to SIMD for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration
/// - AVX2: 8 f32s or 4 f64s per iteration
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn sigmoid_kernel<T: Element>(a: *const T, out: *mut T, len: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use super::super::simd::activations;

        match T::DTYPE {
            DType::F32 => {
                activations::sigmoid_f32(a as *const f32, out as *mut f32, len);
                return;
            }
            DType::F64 => {
                activations::sigmoid_f64(a as *const f64, out as *mut f64, len);
                return;
            }
            _ => {}
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use super::super::simd::activations;

        match T::DTYPE {
            DType::F32 => {
                activations::sigmoid_f32(a as *const f32, out as *mut f32, len);
                return;
            }
            DType::F64 => {
                activations::sigmoid_f64(a as *const f64, out as *mut f64, len);
                return;
            }
            _ => {}
        }
    }

    sigmoid_scalar(a, out, len);
}

/// Scalar sigmoid for all Element types
#[inline]
unsafe fn sigmoid_scalar<T: Element>(a: *const T, out: *mut T, len: usize) {
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
    #[cfg(target_arch = "x86_64")]
    {
        use super::super::simd::activations;

        match T::DTYPE {
            DType::F32 => {
                activations::silu_f32(a as *const f32, out as *mut f32, len);
                return;
            }
            DType::F64 => {
                activations::silu_f64(a as *const f64, out as *mut f64, len);
                return;
            }
            _ => {}
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use super::super::simd::activations;

        match T::DTYPE {
            DType::F32 => {
                activations::silu_f32(a as *const f32, out as *mut f32, len);
                return;
            }
            DType::F64 => {
                activations::silu_f64(a as *const f64, out as *mut f64, len);
                return;
            }
            _ => {}
        }
    }

    silu_scalar(a, out, len);
}

/// Scalar SiLU for all Element types
#[inline]
unsafe fn silu_scalar<T: Element>(a: *const T, out: *mut T, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let x = a_slice[i].to_f64();
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
    #[cfg(target_arch = "x86_64")]
    {
        use super::super::simd::activations;

        match T::DTYPE {
            DType::F32 => {
                activations::gelu_f32(a as *const f32, out as *mut f32, len);
                return;
            }
            DType::F64 => {
                activations::gelu_f64(a as *const f64, out as *mut f64, len);
                return;
            }
            _ => {}
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use super::super::simd::activations;

        match T::DTYPE {
            DType::F32 => {
                activations::gelu_f32(a as *const f32, out as *mut f32, len);
                return;
            }
            DType::F64 => {
                activations::gelu_f64(a as *const f64, out as *mut f64, len);
                return;
            }
            _ => {}
        }
    }

    gelu_scalar(a, out, len);
}

/// Scalar GELU for all Element types
#[inline]
unsafe fn gelu_scalar<T: Element>(a: *const T, out: *mut T, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
    const TANH_COEF: f64 = 0.044715;

    for i in 0..len {
        let x = a_slice[i].to_f64();
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
    #[cfg(target_arch = "x86_64")]
    {
        use super::super::simd::activations;

        match T::DTYPE {
            DType::F32 => {
                activations::leaky_relu_f32(
                    a as *const f32,
                    out as *mut f32,
                    len,
                    negative_slope as f32,
                );
                return;
            }
            DType::F64 => {
                activations::leaky_relu_f64(a as *const f64, out as *mut f64, len, negative_slope);
                return;
            }
            _ => {}
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use super::super::simd::activations;

        match T::DTYPE {
            DType::F32 => {
                activations::leaky_relu_f32(
                    a as *const f32,
                    out as *mut f32,
                    len,
                    negative_slope as f32,
                );
                return;
            }
            DType::F64 => {
                activations::leaky_relu_f64(a as *const f64, out as *mut f64, len, negative_slope);
                return;
            }
            _ => {}
        }
    }

    leaky_relu_scalar(a, out, len, negative_slope);
}

/// Scalar Leaky ReLU for all Element types
#[inline]
unsafe fn leaky_relu_scalar<T: Element>(a: *const T, out: *mut T, len: usize, negative_slope: f64) {
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
    #[cfg(target_arch = "x86_64")]
    {
        use super::super::simd::activations;

        match T::DTYPE {
            DType::F32 => {
                activations::elu_f32(a as *const f32, out as *mut f32, len, alpha as f32);
                return;
            }
            DType::F64 => {
                activations::elu_f64(a as *const f64, out as *mut f64, len, alpha);
                return;
            }
            _ => {}
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use super::super::simd::activations;

        match T::DTYPE {
            DType::F32 => {
                activations::elu_f32(a as *const f32, out as *mut f32, len, alpha as f32);
                return;
            }
            DType::F64 => {
                activations::elu_f64(a as *const f64, out as *mut f64, len, alpha);
                return;
            }
            _ => {}
        }
    }

    elu_scalar(a, out, len, alpha);
}

/// Scalar ELU for all Element types
#[inline]
unsafe fn elu_scalar<T: Element>(a: *const T, out: *mut T, len: usize, alpha: f64) {
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
