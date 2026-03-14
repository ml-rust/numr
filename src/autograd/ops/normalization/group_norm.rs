//! Backward implementation for Group Normalization

use crate::autograd::GradFn;
use crate::autograd::var::Var;
use crate::error::Result;
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, TensorOps, UnaryOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Backward for Group Normalization.
///
/// Input shape: `[B, C, *spatial]`. Normalizes over (C/G, *spatial) per group.
///
/// Gradients:
/// - d_input: similar to layer_norm but per-group
/// - d_weight = sum(grad_out * x_norm, batch_and_spatial_dims)
/// - d_bias = sum(grad_out, batch_and_spatial_dims)
pub struct GroupNormBackward<R: Runtime> {
    input_ids: [TensorId; 3], // [input, weight, bias]
    saved_input: Tensor<R>,
    saved_weight: Tensor<R>,
    num_groups: usize,
    eps: f32,
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 3],
}

impl<R: Runtime> GroupNormBackward<R> {
    /// Create a new GroupNormBackward
    pub fn new(
        input_id: TensorId,
        weight_id: TensorId,
        bias_id: TensorId,
        input: Tensor<R>,
        weight: Tensor<R>,
        num_groups: usize,
        eps: f32,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
        weight_grad_fn: Option<Arc<dyn GradFn<R>>>,
        bias_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [input_id, weight_id, bias_id],
            saved_input: input,
            saved_weight: weight,
            num_groups,
            eps,
            input_grad_fns: [input_grad_fn, weight_grad_fn, bias_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for GroupNormBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + BinaryOps<R> + UnaryOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let input = &self.saved_input;
        let weight = &self.saved_weight;
        let shape = input.shape();
        let batch = shape[0];
        let channels = shape[1];
        let cpg = channels / self.num_groups;
        let spatial: usize = shape[2..].iter().product::<usize>().max(1);
        let group_size = cpg * spatial;

        // Flatten to [B, G, C/G * spatial] for per-group normalization
        let flat_shape = [batch, self.num_groups, group_size];
        let input_flat = input.reshape(&flat_shape)?;
        let grad_flat = grad_output.reshape(&flat_shape)?;

        // Per-group mean and variance: reduce over dim 2
        let mu = client.mean(&input_flat, &[2], true)?;
        let x_centered = client.sub(&input_flat, &mu)?;
        let x_sq = client.mul(&x_centered, &x_centered)?;
        let variance = client.mean(&x_sq, &[2], true)?;
        let var_eps = client.add_scalar(&variance, self.eps as f64)?;
        let std = client.sqrt(&var_eps)?;
        let rstd = client.recip(&std)?;
        let x_norm_flat = client.mul(&x_centered, &rstd)?;

        // Reshape weight [C] → [1, G, cpg, 1] → broadcast → [1, G, cpg, spatial] → [1, G, group_size]
        let weight_4d = weight.reshape(&[1, self.num_groups, cpg, 1])?;
        let weight_bcast = weight_4d
            .broadcast_to(&[1, self.num_groups, cpg, spatial])?
            .contiguous();
        let weight_flat = weight_bcast.reshape(&[1, self.num_groups, group_size])?;

        // d_input (per-group layer norm backward)
        let gw = client.mul(&grad_flat, &weight_flat)?;
        let mean_gw = client.mean(&gw, &[2], true)?;
        let gw_xn = client.mul(&gw, &x_norm_flat)?;
        let mean_gw_xn = client.mean(&gw_xn, &[2], true)?;
        let xn_correction = client.mul(&x_norm_flat, &mean_gw_xn)?;
        let inner = client.sub(&gw, &mean_gw)?;
        let inner = client.sub(&inner, &xn_correction)?;
        let d_input_flat = client.mul(&inner, &rstd)?;
        let d_input = d_input_flat.reshape(shape)?;

        // x_norm reshaped back to [B, C, spatial]
        let x_norm_bcs = x_norm_flat.reshape(&[batch, channels, spatial])?;
        let grad_bcs = grad_output.reshape(&[batch, channels, spatial])?;

        // d_weight = sum(grad * x_norm, dims=[0, 2]) → [C]
        let gxn = client.mul(&grad_bcs, &x_norm_bcs)?;
        let d_weight = client.sum(&gxn, &[0, 2], false)?;

        // d_bias = sum(grad, dims=[0, 2]) → [C]
        let d_bias = client.sum(&grad_bcs, &[0, 2], false)?;

        Ok(vec![Some(d_input), Some(d_weight), Some(d_bias)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
    {
        // For higher-order gradients, fall back to tensor backward wrapped in Var
        let grads = self.backward(grad_output.tensor())?;
        Ok(grads
            .into_iter()
            .map(|g| g.map(|t| Var::new(t, false)))
            .collect())
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_input)
    }

    fn name(&self) -> &'static str {
        "GroupNormBackward"
    }
}
