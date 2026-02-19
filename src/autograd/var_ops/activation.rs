//! Activation operations (relu, sigmoid, softmax)

use super::ops::*;
use crate::autograd::Var;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ActivationOps, CompareOps, ReduceOps, ScalarOps, TensorOps, UnaryOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// ReLU: z = max(0, a)
pub fn var_relu<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + CompareOps<R>,
    R::Client: TensorOps<R> + CompareOps<R>,
{
    let output = client.relu(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = ReluBackward::<R>::new(a.id(), a.tensor().clone(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Sigmoid: z = 1 / (1 + exp(-a))
pub fn var_sigmoid<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.sigmoid(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = SigmoidBackward::<R>::new(a.id(), output.clone(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// SiLU (Swish) activation: `z = a * sigmoid(a)`
///
/// A smooth, non-monotonic activation function popular in modern architectures
/// (e.g., SwiGLU in LLaMA). Often preferred over ReLU for its non-zero gradient
/// at negative inputs.
///
/// Gradient: `dz/da = sigmoid(a) * (1 + a - silu(a))`
pub fn var_silu<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ActivationOps<R> + ScalarOps<R>,
    R::Client: TensorOps<R> + ActivationOps<R> + ScalarOps<R>,
{
    let output = client.silu(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = SiluBackward::<R>::new(
            a.id(),
            a.tensor().clone(),
            output.clone(),
            a.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Softplus: `z = log(1 + exp(a))`
///
/// A smooth, always-positive approximation to ReLU. Used in Mamba2 for dt
/// (step size) processing via `softplus(dt_proj(x)) + dt_bias`.
///
/// Computed via the numerically stable form `relu(a) + log(1 + exp(-|a|))`
/// to avoid overflow for large positive inputs.
///
/// Gradient: `dz/da = sigmoid(a)`
pub fn var_softplus<R, C>(a: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ActivationOps<R>,
    R::Client: TensorOps<R> + ActivationOps<R>,
{
    let output = client.softplus(a.tensor())?;

    if a.requires_grad() {
        let grad_fn = SoftplusBackward::<R>::new(a.id(), a.tensor().clone(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Softmax along dimension: z_i = exp(a_i) / sum(exp(a))
pub fn var_softmax<R, C>(a: &Var<R>, dim: isize, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let output = client.softmax(a.tensor(), dim)?;

    if a.requires_grad() {
        let grad_fn = SoftmaxBackward::<R>::new(a.id(), output.clone(), dim, a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Log-softmax along dimension: z = log(softmax(a, dim))
pub fn var_log_softmax<R, C>(a: &Var<R>, dim: isize, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ActivationOps<R>,
    R::Client: TensorOps<R> + UnaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let output = client.log_softmax(a.tensor(), dim)?;

    if a.requires_grad() {
        let grad_fn =
            LogSoftmaxBackward::<R>::new(a.id(), output.clone(), dim, a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}
