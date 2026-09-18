//! Generic implementations of composite normalization operations.

use crate::error::Result;
use crate::ops::traits::{BinaryOps, ReduceOps, UnaryOps, UtilityOps};
use crate::runtime::{Runtime, normalize_dim};
use crate::tensor::Tensor;

/// L2 normalization along `dim`: `x / max(sqrt(sum(x^2, dim)), eps)`.
///
/// `eps` floors the norm, so a zero vector maps to zero. Same formula as
/// `ggml_l2_norm`. All backends delegate here.
pub fn l2_normalize_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    dim: isize,
    eps: f32,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: BinaryOps<R> + UnaryOps<R> + ReduceOps<R> + UtilityOps<R>,
{
    let dim = normalize_dim(dim, input.ndim())?;
    let squared = client.square(input)?;
    let sum_sq = client.sum(&squared, &[dim], true)?;
    let norm = client.sqrt(&sum_sq)?;
    let floored = client.clamp(&norm, f64::from(eps), f64::INFINITY)?;
    client.div(input, &floored)
}
