//! Snake activation kernels: `y = x + sin(alpha * x)^2 / (beta + eps)`.
//!
//! `x`, `out` and `grad` are contiguous `[outer, channels, inner]` buffers;
//! `alpha` and `beta` hold one value per channel. Every element is computed in
//! a [`FloatAcc`]: `f64` for F64 input, `f32` for every other float, so F16 and
//! BF16 round once at the store instead of at every intermediate. The
//! per-channel parameter gradients accumulate in `f64` on the host in a fixed
//! `outer`-then-`inner` order, so a repeated call returns identical bits.

use super::super::wide_acc::FloatAcc;
use crate::dtype::{DType, Element};

/// The arithmetic the Snake kernels need beyond [`FloatAcc`]: `sin` and a
/// round trip through `f64` for the epsilon and the host-side sums.
pub trait SnakeAcc: FloatAcc {
    /// Multiplicative identity.
    const ONE: Self;
    /// Widen an `f64` scalar into the accumulator.
    fn from_f64(v: f64) -> Self;
    /// Narrow the accumulator to `f64`.
    fn to_f64(self) -> f64;
    /// `sin(self)`.
    fn sin(self) -> Self;
    /// `self / other`.
    fn acc_div(self, other: Self) -> Self;
}

impl SnakeAcc for f32 {
    const ONE: Self = 1.0;
    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline]
    fn sin(self) -> Self {
        f32::sin(self)
    }
    #[inline]
    fn acc_div(self, other: Self) -> Self {
        self / other
    }
}

impl SnakeAcc for f64 {
    const ONE: Self = 1.0;
    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline]
    fn acc_div(self, other: Self) -> Self {
        self / other
    }
}

/// Snake forward over a contiguous `[outer, channels, inner]` buffer.
///
/// # Safety
///
/// - `x` and `out` point to `outer * channels * inner` elements.
/// - `alpha` and `beta` point to `channels` elements.
pub unsafe fn snake_beta_kernel<T: Element>(
    x: *const T,
    alpha: *const T,
    beta: *const T,
    out: *mut T,
    outer: usize,
    channels: usize,
    inner: usize,
    eps: f64,
) {
    if T::DTYPE == DType::F64 {
        snake_beta_fwd_acc::<T, f64>(x, alpha, beta, out, outer, channels, inner, eps);
    } else {
        snake_beta_fwd_acc::<T, f32>(x, alpha, beta, out, outer, channels, inner, eps);
    }
}

/// [`snake_beta_kernel`] with the accumulator fixed by the caller.
///
/// # Safety
///
/// Same as [`snake_beta_kernel`].
#[allow(clippy::too_many_arguments)]
unsafe fn snake_beta_fwd_acc<T: Element, A: SnakeAcc>(
    x: *const T,
    alpha: *const T,
    beta: *const T,
    out: *mut T,
    outer: usize,
    channels: usize,
    inner: usize,
    eps: f64,
) {
    let eps = A::from_f64(eps);
    for o in 0..outer {
        for c in 0..channels {
            let a = A::from_elem(*alpha.add(c));
            let inv = A::ONE.acc_div(A::from_elem(*beta.add(c)).acc_add(eps));
            let base = (o * channels + c) * inner;
            for i in 0..inner {
                let xv = A::from_elem(*x.add(base + i));
                let s = a.acc_mul(xv).sin();
                *out.add(base + i) = xv.acc_add(s.acc_mul(s).acc_mul(inv)).to_elem();
            }
        }
    }
}

/// Snake backward over a contiguous `[outer, channels, inner]` buffer.
///
/// Writes `d_x` per element and the per-channel `d_alpha` / `d_beta` sums.
///
/// # Safety
///
/// - `grad`, `x` and `d_x` point to `outer * channels * inner` elements.
/// - `alpha`, `beta`, `d_alpha` and `d_beta` point to `channels` elements.
#[allow(clippy::too_many_arguments)]
pub unsafe fn snake_beta_bwd_kernel<T: Element>(
    grad: *const T,
    x: *const T,
    alpha: *const T,
    beta: *const T,
    d_x: *mut T,
    d_alpha: *mut T,
    d_beta: *mut T,
    outer: usize,
    channels: usize,
    inner: usize,
    eps: f64,
) {
    if T::DTYPE == DType::F64 {
        snake_beta_bwd_acc::<T, f64>(
            grad, x, alpha, beta, d_x, d_alpha, d_beta, outer, channels, inner, eps,
        );
    } else {
        snake_beta_bwd_acc::<T, f32>(
            grad, x, alpha, beta, d_x, d_alpha, d_beta, outer, channels, inner, eps,
        );
    }
}

/// [`snake_beta_bwd_kernel`] with the accumulator fixed by the caller.
///
/// # Safety
///
/// Same as [`snake_beta_bwd_kernel`].
#[allow(clippy::too_many_arguments)]
unsafe fn snake_beta_bwd_acc<T: Element, A: SnakeAcc>(
    grad: *const T,
    x: *const T,
    alpha: *const T,
    beta: *const T,
    d_x: *mut T,
    d_alpha: *mut T,
    d_beta: *mut T,
    outer: usize,
    channels: usize,
    inner: usize,
    eps: f64,
) {
    let eps = A::from_f64(eps);
    let two = A::from_f64(2.0);
    // Host-side channel sums in f64, fixed order: outer, then inner.
    let mut sum_alpha = vec![0.0f64; channels];
    let mut sum_beta = vec![0.0f64; channels];
    for o in 0..outer {
        for c in 0..channels {
            let a = A::from_elem(*alpha.add(c));
            let inv = A::ONE.acc_div(A::from_elem(*beta.add(c)).acc_add(eps));
            let inv_sq = inv.acc_mul(inv);
            let base = (o * channels + c) * inner;
            let mut acc_alpha = 0.0f64;
            let mut acc_beta = 0.0f64;
            for i in 0..inner {
                let xv = A::from_elem(*x.add(base + i));
                let g = A::from_elem(*grad.add(base + i));
                let ax = a.acc_mul(xv);
                let s = ax.sin();
                let s2 = two.acc_mul(ax).sin();
                *d_x.add(base + i) = g
                    .acc_mul(A::ONE.acc_add(a.acc_mul(s2).acc_mul(inv)))
                    .to_elem();
                acc_alpha += g.acc_mul(xv).acc_mul(s2).acc_mul(inv).to_f64();
                acc_beta -= g.acc_mul(s).acc_mul(s).acc_mul(inv_sq).to_f64();
            }
            sum_alpha[c] += acc_alpha;
            sum_beta[c] += acc_beta;
        }
    }
    for c in 0..channels {
        *d_alpha.add(c) = T::from_f64(sum_alpha[c]);
        *d_beta.add(c) = T::from_f64(sum_beta[c]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_matches_scalar_formula_in_f64() {
        let x = [0.3f64, -1.2, 2.5, 0.0, 4.0, -0.7];
        let alpha = [1.5f64, 0.25];
        let beta = [0.8f64, 3.0];
        let eps = 1e-9;
        let mut out = [0.0f64; 6];
        unsafe {
            snake_beta_kernel(
                x.as_ptr(),
                alpha.as_ptr(),
                beta.as_ptr(),
                out.as_mut_ptr(),
                1,
                2,
                3,
                eps,
            );
        }
        for (i, (&got, &xv)) in out.iter().zip(x.iter()).enumerate() {
            let c = i / 3;
            let want = xv + (alpha[c] * xv).sin().powi(2) / (beta[c] + eps);
            assert!((got - want).abs() < 1e-14, "index {i}: {got} vs {want}");
        }
    }

    #[test]
    fn backward_sums_channels_across_outer_and_inner() {
        // [outer=2, channels=1, inner=2]: all four elements feed channel 0.
        let x = [0.5f64, -0.25, 1.0, 2.0];
        let grad = [1.0f64, 2.0, -1.0, 0.5];
        let alpha = [1.1f64];
        let beta = [0.9f64];
        let eps = 1e-9;
        let mut d_x = [0.0f64; 4];
        let mut d_alpha = [0.0f64];
        let mut d_beta = [0.0f64];
        unsafe {
            snake_beta_bwd_kernel(
                grad.as_ptr(),
                x.as_ptr(),
                alpha.as_ptr(),
                beta.as_ptr(),
                d_x.as_mut_ptr(),
                d_alpha.as_mut_ptr(),
                d_beta.as_mut_ptr(),
                2,
                1,
                2,
                eps,
            );
        }
        let inv = 1.0 / (beta[0] + eps);
        let mut want_alpha = 0.0;
        let mut want_beta = 0.0;
        for i in 0..4 {
            let s2 = (2.0 * alpha[0] * x[i]).sin();
            let s = (alpha[0] * x[i]).sin();
            let want_dx = grad[i] * (1.0 + alpha[0] * s2 * inv);
            assert!((d_x[i] - want_dx).abs() < 1e-14, "d_x[{i}]");
            want_alpha += grad[i] * x[i] * s2 * inv;
            want_beta -= grad[i] * s * s * inv * inv;
        }
        assert!((d_alpha[0] - want_alpha).abs() < 1e-14);
        assert!((d_beta[0] - want_beta).abs() < 1e-14);
    }
}
