//! Snake activation helpers for CPU tensors.
//!
//! Both entry points validate through
//! [`validate_snake_beta`](crate::ops::activation_common::validate_snake_beta),
//! make `x` (and `grad`) contiguous, and hand the `[outer, C, inner]` view to
//! the kernels in `kernels::unary::snake`.

use super::super::kernels;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::error::Result;
use crate::ops::activation_common::validate_snake_beta;
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// `y = x + sin(alpha * x)^2 / (beta + eps)`, channel axis `dim`.
pub fn snake_beta_impl(
    client: &CpuClient,
    x: &Tensor<CpuRuntime>,
    alpha: &Tensor<CpuRuntime>,
    beta: &Tensor<CpuRuntime>,
    dim: isize,
    eps: f64,
) -> Result<Tensor<CpuRuntime>> {
    let geom = validate_snake_beta(x, alpha, beta, dim, eps)?;
    let dtype = x.dtype();
    let x_contig = ensure_contiguous(x)?;
    let alpha_contig = ensure_contiguous(alpha)?;
    let beta_contig = ensure_contiguous(beta)?;
    let out = Tensor::<CpuRuntime>::empty(x.shape(), dtype, &client.device)?;

    let x_ptr = x_contig.ptr();
    let alpha_ptr = alpha_contig.ptr();
    let beta_ptr = beta_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::snake_beta_kernel::<T>(
                x_ptr as *const T,
                alpha_ptr as *const T,
                beta_ptr as *const T,
                out_ptr as *mut T,
                geom.outer,
                geom.channels,
                geom.inner,
                eps,
            );
        }
    }, "snake_beta");

    Ok(out)
}

/// `(d_x, d_alpha, d_beta)` for [`snake_beta_impl`].
#[allow(clippy::type_complexity)]
pub fn snake_beta_bwd_impl(
    client: &CpuClient,
    grad: &Tensor<CpuRuntime>,
    x: &Tensor<CpuRuntime>,
    alpha: &Tensor<CpuRuntime>,
    beta: &Tensor<CpuRuntime>,
    dim: isize,
    eps: f64,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let geom = validate_snake_beta(x, alpha, beta, dim, eps)?;
    let dtype = x.dtype();
    if grad.shape() != x.shape() {
        return Err(crate::error::Error::ShapeMismatch {
            expected: x.shape().to_vec(),
            got: grad.shape().to_vec(),
        });
    }
    if grad.dtype() != dtype {
        return Err(crate::error::Error::DTypeMismatch {
            lhs: dtype,
            rhs: grad.dtype(),
        });
    }
    let grad_contig = ensure_contiguous(grad)?;
    let x_contig = ensure_contiguous(x)?;
    let alpha_contig = ensure_contiguous(alpha)?;
    let beta_contig = ensure_contiguous(beta)?;
    let d_x = Tensor::<CpuRuntime>::empty(x.shape(), dtype, &client.device)?;
    let d_alpha = Tensor::<CpuRuntime>::empty(alpha.shape(), dtype, &client.device)?;
    let d_beta = Tensor::<CpuRuntime>::empty(beta.shape(), dtype, &client.device)?;

    let grad_ptr = grad_contig.ptr();
    let x_ptr = x_contig.ptr();
    let alpha_ptr = alpha_contig.ptr();
    let beta_ptr = beta_contig.ptr();
    let d_x_ptr = d_x.ptr();
    let d_alpha_ptr = d_alpha.ptr();
    let d_beta_ptr = d_beta.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::snake_beta_bwd_kernel::<T>(
                grad_ptr as *const T,
                x_ptr as *const T,
                alpha_ptr as *const T,
                beta_ptr as *const T,
                d_x_ptr as *mut T,
                d_alpha_ptr as *mut T,
                d_beta_ptr as *mut T,
                geom.outer,
                geom.channels,
                geom.inner,
                eps,
            );
        }
    }, "snake_beta_bwd");

    Ok((d_x, d_alpha, d_beta))
}
