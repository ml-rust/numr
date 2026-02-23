//! Fused activation-multiplication kernels
//!
//! Each function computes `activation(a) * b` element-wise with automatic SIMD dispatch.
//! Fusing saves one full memory pass compared to separate activation + multiply.

use crate::dtype::{DType, Element};

/// Fused SiLU-Mul: `silu(a) * b = (a / (1 + exp(-a))) * b`
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn silu_mul_kernel<T: Element>(a: *const T, b: *const T, out: *mut T, len: usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::fused_activation_mul;

        match T::DTYPE {
            DType::F32 => {
                fused_activation_mul::silu_mul_f32(
                    a as *const f32,
                    b as *const f32,
                    out as *mut f32,
                    len,
                );
                return;
            }
            DType::F64 => {
                fused_activation_mul::silu_mul_f64(
                    a as *const f64,
                    b as *const f64,
                    out as *mut f64,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                fused_activation_mul::silu_mul_f16(
                    a as *const half::f16,
                    b as *const half::f16,
                    out as *mut half::f16,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                fused_activation_mul::silu_mul_bf16(
                    a as *const half::bf16,
                    b as *const half::bf16,
                    out as *mut half::bf16,
                    len,
                );
                return;
            }
            _ => {}
        }
    }

    fused_scalar(a, b, out, len, |x| x / (1.0 + (-x).exp()));
}

/// Fused GELU-Mul: `gelu(a) * b`
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn gelu_mul_kernel<T: Element>(a: *const T, b: *const T, out: *mut T, len: usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::fused_activation_mul;

        match T::DTYPE {
            DType::F32 => {
                fused_activation_mul::gelu_mul_f32(
                    a as *const f32,
                    b as *const f32,
                    out as *mut f32,
                    len,
                );
                return;
            }
            DType::F64 => {
                fused_activation_mul::gelu_mul_f64(
                    a as *const f64,
                    b as *const f64,
                    out as *mut f64,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                fused_activation_mul::gelu_mul_f16(
                    a as *const half::f16,
                    b as *const half::f16,
                    out as *mut half::f16,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                fused_activation_mul::gelu_mul_bf16(
                    a as *const half::bf16,
                    b as *const half::bf16,
                    out as *mut half::bf16,
                    len,
                );
                return;
            }
            _ => {}
        }
    }

    const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
    const TANH_COEF: f64 = 0.044715;
    fused_scalar(a, b, out, len, |x| {
        let inner = SQRT_2_OVER_PI * (x + TANH_COEF * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    });
}

/// Fused ReLU-Mul: `relu(a) * b = max(0, a) * b`
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn relu_mul_kernel<T: Element>(a: *const T, b: *const T, out: *mut T, len: usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::fused_activation_mul;

        match T::DTYPE {
            DType::F32 => {
                fused_activation_mul::relu_mul_f32(
                    a as *const f32,
                    b as *const f32,
                    out as *mut f32,
                    len,
                );
                return;
            }
            DType::F64 => {
                fused_activation_mul::relu_mul_f64(
                    a as *const f64,
                    b as *const f64,
                    out as *mut f64,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                fused_activation_mul::relu_mul_f16(
                    a as *const half::f16,
                    b as *const half::f16,
                    out as *mut half::f16,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                fused_activation_mul::relu_mul_bf16(
                    a as *const half::bf16,
                    b as *const half::bf16,
                    out as *mut half::bf16,
                    len,
                );
                return;
            }
            _ => {}
        }
    }

    fused_scalar(a, b, out, len, |x| if x > 0.0 { x } else { 0.0 });
}

/// Fused Sigmoid-Mul: `sigmoid(a) * b = (1 / (1 + exp(-a))) * b`
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn sigmoid_mul_kernel<T: Element>(a: *const T, b: *const T, out: *mut T, len: usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::fused_activation_mul;

        match T::DTYPE {
            DType::F32 => {
                fused_activation_mul::sigmoid_mul_f32(
                    a as *const f32,
                    b as *const f32,
                    out as *mut f32,
                    len,
                );
                return;
            }
            DType::F64 => {
                fused_activation_mul::sigmoid_mul_f64(
                    a as *const f64,
                    b as *const f64,
                    out as *mut f64,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                fused_activation_mul::sigmoid_mul_f16(
                    a as *const half::f16,
                    b as *const half::f16,
                    out as *mut half::f16,
                    len,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                fused_activation_mul::sigmoid_mul_bf16(
                    a as *const half::bf16,
                    b as *const half::bf16,
                    out as *mut half::bf16,
                    len,
                );
                return;
            }
            _ => {}
        }
    }

    fused_scalar(a, b, out, len, |x| 1.0 / (1.0 + (-x).exp()));
}

/// Generic scalar fallback for fused activation-mul: `activation(a[i]) * b[i]`
#[inline]
unsafe fn fused_scalar<T: Element, F: Fn(f64) -> f64>(
    a: *const T,
    b: *const T,
    out: *mut T,
    len: usize,
    activation: F,
) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let b_slice = std::slice::from_raw_parts(b, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        let x = a_slice[i].to_f64();
        let y = b_slice[i].to_f64();
        out_slice[i] = T::from_f64(activation(x) * y);
    }
}
