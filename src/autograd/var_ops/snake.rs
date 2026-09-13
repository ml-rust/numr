//! Snake activation with gradient support: `y = x + sin(alpha * x)^2 / (beta + eps)`.
//!
//! Forward runs the fused `snake_beta` kernel. First-order backward runs the
//! fused `snake_beta_bwd` kernel, which returns `d_x`, `d_alpha` and `d_beta`
//! together. Second-order backward treats the saved `x`, `alpha` and `beta` as
//! constants (the convention the other activation backwards follow) and builds
//! the three gradients from `var_mul` / `var_sum`, so the dependence on the
//! incoming gradient stays on the graph.

use crate::autograd::var_ops::reduce::var_sum;
use crate::autograd::var_ops::var_mul;
use crate::autograd::{GradFn, Var};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::activation_common::validate_snake_beta;
use crate::ops::{ActivationOps, BinaryOps, ScalarOps, TensorOps, UnaryOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Snake activation along `dim` with per-channel LINEAR-scale `alpha` and `beta`.
///
/// Gradients flow to `x`, `alpha` and `beta`. Plain Snake is `beta == alpha`:
/// pass the same `Var` twice and the two parameter gradients are both
/// accumulated onto it.
pub fn var_snake_beta<R, C>(
    x: &Var<R>,
    alpha: &Var<R>,
    beta: &Var<R>,
    dim: isize,
    eps: f64,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ActivationOps<R>,
    R::Client: TensorOps<R> + ActivationOps<R> + ScalarOps<R>,
{
    let output = client.snake_beta(x.tensor(), alpha.tensor(), beta.tensor(), dim, eps)?;

    if x.requires_grad() || alpha.requires_grad() || beta.requires_grad() {
        let grad_fn = SnakeBetaBackward::<R> {
            input_ids: [x.id(), alpha.id(), beta.id()],
            saved: [
                x.tensor().clone(),
                alpha.tensor().clone(),
                beta.tensor().clone(),
            ],
            dim,
            eps,
            input_grad_fns: [
                x.grad_fn().cloned(),
                alpha.grad_fn().cloned(),
                beta.grad_fn().cloned(),
            ],
        };
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Backward for `snake_beta`: inputs `[x, alpha, beta]`.
pub struct SnakeBetaBackward<R: Runtime> {
    input_ids: [TensorId; 3],
    /// `[x, alpha, beta]`.
    saved: [Tensor<R>; 3],
    dim: isize,
    eps: f64,
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 3],
}

impl<R: Runtime<DType = DType>> GradFn<R> for SnakeBetaBackward<R>
where
    R::Client: TensorOps<R> + ActivationOps<R> + ScalarOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // One fused kernel returns all three gradients; no slot has a cost of
        // its own to skip.
        let client = R::default_client(grad_output.device());
        let [x, alpha, beta] = &self.saved;
        let (d_x, d_alpha, d_beta) =
            client.snake_beta_bwd(grad_output, x, alpha, beta, self.dim, self.eps)?;
        Ok(vec![Some(d_x), Some(d_alpha), Some(d_beta)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ActivationOps<R> + ScalarOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());
        let [x, alpha, beta] = &self.saved;
        let geom = validate_snake_beta(x, alpha, beta, self.dim, self.eps)?;

        // Broadcast shape for the per-channel parameters: 1 everywhere except
        // the channel axis.
        let ndim = x.shape().len();
        let dim_idx = geom_dim(ndim, self.dim);
        let mut param_shape = vec![1usize; ndim];
        param_shape[dim_idx] = geom.channels;
        let alpha_b = alpha.reshape(&param_shape)?;
        let beta_b = beta.reshape(&param_shape)?;

        // Constants w.r.t. grad_output:
        //   k_x     = 1 + alpha * sin(2 alpha x) / (beta + eps)
        //   k_alpha = x * sin(2 alpha x) / (beta + eps)
        //   k_beta  = -sin(alpha x)^2 / (beta + eps)^2
        let inv = client.recip(&client.add_scalar(&beta_b, self.eps)?)?;
        let ax = client.mul(x, &alpha_b)?;
        let s2 = client.sin(&client.mul_scalar(&ax, 2.0)?)?;
        let s = client.sin(&ax)?;
        let s2_inv = client.mul(&s2, &inv)?;
        let k_x = client.add_scalar(&client.mul(&s2_inv, &alpha_b)?, 1.0)?;
        let k_alpha = client.mul(&s2_inv, x)?;
        let s_sq = client.mul(&s, &s)?;
        let k_beta = client.neg(&client.mul(&client.mul(&s_sq, &inv)?, &inv)?)?;

        let reduce_dims: Vec<usize> = (0..ndim).filter(|&d| d != dim_idx).collect();
        let d_x = var_mul(grad_output, &Var::new(k_x, false), &client)?;
        let d_alpha_full = var_mul(grad_output, &Var::new(k_alpha, false), &client)?;
        let d_beta_full = var_mul(grad_output, &Var::new(k_beta, false), &client)?;
        let d_alpha = var_sum(&d_alpha_full, &reduce_dims, false, &client)?;
        let d_beta = var_sum(&d_beta_full, &reduce_dims, false, &client)?;
        Ok(vec![Some(d_x), Some(d_alpha), Some(d_beta)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        &self.saved
    }

    fn name(&self) -> &'static str {
        "SnakeBetaBackward"
    }
}

/// Normalised channel axis; `dim` was validated by the forward call.
fn geom_dim(ndim: usize, dim: isize) -> usize {
    if dim >= 0 {
        dim as usize
    } else {
        (ndim as isize + dim) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    fn setup() -> (
        <CpuRuntime as Runtime>::Client,
        CpuDevice,
        Var<CpuRuntime>,
        Var<CpuRuntime>,
        Var<CpuRuntime>,
    ) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &[0.3f64, -1.2, 2.5, 0.0, 4.0, -0.7, 1.1, 0.9],
                &[2, 2, 2],
                &device,
            )
            .unwrap(),
            true,
        );
        let alpha = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.5f64, 0.25], &[2], &device).unwrap(),
            true,
        );
        let beta = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.8f64, 3.0], &[2], &device).unwrap(),
            true,
        );
        (client, device, x, alpha, beta)
    }

    /// Central finite differences of `sum(snake_beta)` for one parameter tensor.
    fn numeric_grad(
        client: &<CpuRuntime as Runtime>::Client,
        device: &CpuDevice,
        which: usize,
        base: [&Tensor<CpuRuntime>; 3],
    ) -> Vec<f64> {
        let h = 1e-6;
        let values: Vec<f64> = base[which].to_vec();
        let mut grad = Vec::with_capacity(values.len());
        for i in 0..values.len() {
            let eval = |delta: f64| -> f64 {
                let mut v = values.clone();
                v[i] += delta;
                let t = Tensor::<CpuRuntime>::from_slice(&v, base[which].shape(), device).unwrap();
                let args: Vec<&Tensor<CpuRuntime>> = (0..3)
                    .map(|k| if k == which { &t } else { base[k] })
                    .collect();
                let y = client
                    .snake_beta(args[0], args[1], args[2], 1, 1e-9)
                    .unwrap();
                y.to_vec::<f64>().iter().sum()
            };
            grad.push((eval(h) - eval(-h)) / (2.0 * h));
        }
        grad
    }

    #[test]
    fn forward_matches_client_op() {
        let (client, _device, x, alpha, beta) = setup();
        let y = var_snake_beta(&x, &alpha, &beta, 1, 1e-9, &client).unwrap();
        let want = client
            .snake_beta(x.tensor(), alpha.tensor(), beta.tensor(), 1, 1e-9)
            .unwrap();
        assert_eq!(y.tensor().to_vec::<f64>(), want.to_vec::<f64>());
        assert!(y.requires_grad());
    }

    #[test]
    fn backward_matches_finite_differences_for_all_three_inputs() {
        let (client, device, x, alpha, beta) = setup();
        let y = var_snake_beta(&x, &alpha, &beta, 1, 1e-9, &client).unwrap();
        let loss = var_sum(&y, &[0, 1, 2], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let base = [x.tensor(), alpha.tensor(), beta.tensor()];
        for (which, var) in [&x, &alpha, &beta].into_iter().enumerate() {
            let got: Vec<f64> = grads.get(var.id()).expect("gradient present").to_vec();
            let want = numeric_grad(&client, &device, which, base);
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() < 1e-6,
                    "input {which} element {i}: analytic {g} vs numeric {w}"
                );
            }
        }
    }

    #[test]
    fn shared_alpha_beta_accumulates_both_parameter_gradients() {
        let (client, device, x, alpha, _beta) = setup();
        let y = var_snake_beta(&x, &alpha, &alpha, 1, 1e-9, &client).unwrap();
        let loss = var_sum(&y, &[0, 1, 2], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        let got: Vec<f64> = grads.get(alpha.id()).expect("gradient present").to_vec();

        // d/da of f(a, a) is the sum of both partials.
        let base = [x.tensor(), alpha.tensor(), alpha.tensor()];
        let d_alpha = numeric_grad(&client, &device, 1, base);
        let d_beta = numeric_grad(&client, &device, 2, base);
        for i in 0..got.len() {
            let want = d_alpha[i] + d_beta[i];
            assert!(
                (got[i] - want).abs() < 1e-6,
                "element {i}: analytic {} vs numeric {want}",
                got[i]
            );
        }
    }

    #[test]
    fn no_grad_inputs_produce_a_detached_var() {
        let (client, device, _x, _alpha, _beta) = setup();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.5f32, 1.5], &[2], &device).unwrap(),
            false,
        );
        let p = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap(),
            false,
        );
        let y = var_snake_beta(&x, &p, &p, 0, 1e-9, &client).unwrap();
        assert!(!y.requires_grad());
        assert!(y.grad_fn().is_none());
    }

    #[test]
    fn invalid_arguments_are_rejected() {
        let (client, device, x, alpha, beta) = setup();
        let short = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap(),
            true,
        );
        assert!(var_snake_beta(&x, &short, &beta, 1, 1e-9, &client).is_err());
        assert!(var_snake_beta(&x, &alpha, &beta, 3, 1e-9, &client).is_err());
        assert!(var_snake_beta(&x, &alpha, &beta, 1, -1.0, &client).is_err());
    }
}
