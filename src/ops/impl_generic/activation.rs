//! Generic implementations of composite activation operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::activation::normalize_softmax_dim;
use crate::ops::traits::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, RandomOps, ScalarOps,
};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Generic log_softmax implementation: log_softmax(x, dim) = x - logsumexp(x, dim, keepdim=true)
///
/// This is the canonical algorithm — all backends delegate here.
/// Numerically stable because logsumexp uses the max-subtraction trick internally.
pub fn log_softmax_impl<R, C>(client: &C, a: &Tensor<R>, dim: isize) -> Result<Tensor<R>>
where
    R: Runtime,
    C: BinaryOps<R> + CumulativeOps<R>,
{
    let ndim = a.ndim();
    let dim_idx = normalize_softmax_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

    let lse = client.logsumexp(a, &[dim_idx], true)?;
    client.sub(a, &lse)
}

/// Generic dropout implementation: where(rand > p, x / (1-p), 0)
///
/// During training, randomly zeros elements with probability `p` and scales
/// remaining elements by `1/(1-p)` to preserve expected values.
/// During inference (`training=false`), returns input unchanged.
pub fn dropout_impl<R, C>(client: &C, a: &Tensor<R>, p: f64, training: bool) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RandomOps<R> + CompareOps<R> + ConditionalOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    if !training || p == 0.0 {
        return Ok(a.clone());
    }
    if p >= 1.0 {
        return Ok(Tensor::<R>::zeros(a.shape(), a.dtype(), client.device()));
    }

    // Generate random mask: rand > p means "keep"
    let rand_tensor = client.rand(a.shape(), a.dtype())?;
    let threshold = Tensor::<R>::full_scalar(a.shape(), a.dtype(), p, client.device());
    let mask = client.gt(&rand_tensor, &threshold)?;

    // Scale kept values by 1/(1-p)
    let scale = 1.0 / (1.0 - p);
    let scaled = client.mul_scalar(a, scale)?;

    // Zero out dropped elements
    let zeros = Tensor::<R>::zeros(a.shape(), a.dtype(), client.device());
    client.where_cond(&mask, &scaled, &zeros)
}
