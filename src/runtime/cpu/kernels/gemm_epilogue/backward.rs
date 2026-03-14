//! Backward kernel for GEMM epilogue operations.
//!
//! Computes gradients for `activation(A @ B + bias)`.
//! Accumulation is done in f32 for sub-f32 types (F16, BF16) and in native
//! precision for F32/F64, matching standard ML framework practice.

use crate::dtype::{DType, Element};
use crate::ops::GemmActivation;

/// Float type used for accumulation in backward pass.
///
/// Only f32 and f64 are used as accumulation types. This trait provides
/// the minimal interface needed for the backward kernel to be generic
/// over both precisions.
trait AccFloat:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::AddAssign
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Neg<Output = Self>
    + PartialOrd
{
    fn zero() -> Self;
    fn one() -> Self;
    fn half() -> Self;
    fn from_elem<T: Element>(v: T) -> Self;
    fn to_elem<T: Element>(self) -> T;
    fn tanh(self) -> Self;
    fn exp(self) -> Self;
    fn recip(self) -> Self;
    fn from_f64_const(v: f64) -> Self;
    fn is_finite(self) -> bool;
}

impl AccFloat for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn half() -> Self {
        0.5
    }
    #[inline]
    fn from_elem<T: Element>(v: T) -> Self {
        v.to_f32()
    }
    #[inline]
    fn to_elem<T: Element>(self) -> T {
        T::from_f32(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        f32::tanh(self)
    }
    #[inline]
    fn exp(self) -> Self {
        f32::exp(self)
    }
    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }
    #[inline]
    fn from_f64_const(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
}

impl AccFloat for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn half() -> Self {
        0.5
    }
    #[inline]
    fn from_elem<T: Element>(v: T) -> Self {
        v.to_f64()
    }
    #[inline]
    fn to_elem<T: Element>(self) -> T {
        T::from_f64(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        f64::tanh(self)
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }
    #[inline]
    fn from_f64_const(v: f64) -> Self {
        v
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
}

/// Backward pass for fused matmul + bias + activation.
///
/// Given `output = activation(A @ B + bias)`, computes:
/// - `d_a = (grad * activation'(pre_act)) @ B^T`
/// - `d_b = A^T @ (grad * activation'(pre_act))`
/// - `d_bias = sum(grad * activation'(pre_act), dim=0)`
///
/// where `pre_act = A @ B + bias`.
///
/// # Safety
/// - All pointers must be valid for the specified dimensions
/// - Output pointers must not alias with input pointers
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bias_activation_bwd_kernel<T: Element>(
    grad: *const T,
    a: *const T,
    b: *const T,
    bias: *const T,
    d_a: *mut T,
    d_b: *mut T,
    d_bias: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ld_grad: usize,
    activation: GemmActivation,
) {
    if T::DTYPE == DType::F64 {
        bwd_in::<T, f64>(
            grad, a, b, bias, d_a, d_b, d_bias, m, n, k, lda, ldb, ld_grad, activation,
        );
    } else {
        bwd_in::<T, f32>(
            grad, a, b, bias, d_a, d_b, d_bias, m, n, k, lda, ldb, ld_grad, activation,
        );
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn bwd_in<T: Element, A: AccFloat>(
    grad: *const T,
    a: *const T,
    b: *const T,
    bias: *const T,
    d_a: *mut T,
    d_b: *mut T,
    d_bias: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ld_grad: usize,
    activation: GemmActivation,
) {
    let total = m * n;

    // Step 1: pre_act = A @ B + bias, then grad_pre = grad * activation'(pre_act)
    let mut grad_pre = vec![A::zero(); total];
    for i in 0..m {
        for j in 0..n {
            grad_pre[i * n + j] = A::from_elem(*bias.add(j));
        }
    }
    for i in 0..m {
        for kk in 0..k {
            let a_val: A = A::from_elem(*a.add(i * lda + kk));
            for j in 0..n {
                grad_pre[i * n + j] += a_val * A::from_elem(*b.add(kk * ldb + j));
            }
        }
    }
    for i in 0..total {
        let g: A = A::from_elem(*grad.add((i / n) * ld_grad + (i % n)));
        let deriv = activation_derivative(grad_pre[i], activation);
        // Guard against non-finite derivatives from platform-specific FP edge cases
        let deriv = if deriv.is_finite() { deriv } else { A::zero() };
        grad_pre[i] = g * deriv;
    }

    // Step 2: d_a = grad_pre @ B^T
    let mut d_a_buf = vec![A::zero(); m * k];
    for i in 0..m {
        for j in 0..n {
            let gp = grad_pre[i * n + j];
            for kk in 0..k {
                d_a_buf[i * k + kk] += gp * A::from_elem(*b.add(kk * ldb + j));
            }
        }
    }
    for i in 0..m * k {
        *d_a.add(i) = d_a_buf[i].to_elem::<T>();
    }

    // Step 3: d_b = A^T @ grad_pre
    let mut d_b_buf = vec![A::zero(); k * n];
    for i in 0..m {
        for kk in 0..k {
            let a_val: A = A::from_elem(*a.add(i * lda + kk));
            for j in 0..n {
                d_b_buf[kk * n + j] += a_val * grad_pre[i * n + j];
            }
        }
    }
    for i in 0..k * n {
        *d_b.add(i) = d_b_buf[i].to_elem::<T>();
    }

    // Step 4: d_bias = sum(grad_pre, dim=0)
    let mut d_bias_buf = vec![A::zero(); n];
    for i in 0..m {
        for j in 0..n {
            d_bias_buf[j] += grad_pre[i * n + j];
        }
    }
    for j in 0..n {
        *d_bias.add(j) = d_bias_buf[j].to_elem::<T>();
    }
}

/// Compute activation derivative at the pre-activation value.
fn activation_derivative<A: AccFloat>(pre_act: A, activation: GemmActivation) -> A {
    match activation {
        GemmActivation::None => A::one(),
        GemmActivation::ReLU => {
            if pre_act > A::zero() {
                A::one()
            } else {
                A::zero()
            }
        }
        GemmActivation::GELU => {
            let sqrt_2_over_pi = A::from_f64_const(0.7978845608028654);
            let coef = A::from_f64_const(0.044715);
            let three = A::from_f64_const(3.0);
            let x = pre_act;
            let inner = sqrt_2_over_pi * (x + coef * x * x * x);
            let tanh_val = inner.tanh();
            let sech2 = A::one() - tanh_val * tanh_val;
            let d_inner = sqrt_2_over_pi * (A::one() + three * coef * x * x);
            A::half() * (A::one() + tanh_val) + A::half() * x * sech2 * d_inner
        }
        GemmActivation::SiLU => {
            let sig = (A::one() + (-pre_act).exp()).recip();
            sig + pre_act * sig * (A::one() - sig)
        }
        GemmActivation::Sigmoid => {
            let sig = (A::one() + (-pre_act).exp()).recip();
            sig * (A::one() - sig)
        }
        GemmActivation::Tanh => {
            let t = pre_act.tanh();
            A::one() - t * t
        }
    }
}
