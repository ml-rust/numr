//! Backward implementations for normalization operations
//!
//! Implements gradient computation for rms_norm and layer_norm.

use crate::autograd::GradFn;
use crate::autograd::var::Var;
use crate::autograd::var_ops::{var_mean, var_mul, var_sub, var_sum};
use crate::error::Result;
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, TensorOps, UnaryOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

// ============================================================================
// RmsNormBackward
// ============================================================================

/// Backward for RMS Normalization: y = x / rms(x) * weight
///
/// Where rms(x) = sqrt(mean(x^2, dim=-1) + eps)
///
/// Gradients:
/// - d_input = rstd * (grad_out * weight - x_norm * mean(grad_out * weight * x_norm, dim=-1))
/// - d_weight = sum(grad_out * x_norm, batch_dims)
///
/// Where rstd = 1/rms(x), x_norm = x * rstd
pub struct RmsNormBackward<R: Runtime> {
    input_ids: [TensorId; 2],
    saved_tensors: Vec<Tensor<R>>, // [input, weight]
    eps: f32,
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 2],
}

impl<R: Runtime> RmsNormBackward<R> {
    /// Create a new RmsNormBackward
    pub fn new(
        input_id: TensorId,
        weight_id: TensorId,
        input: Tensor<R>,
        weight: Tensor<R>,
        eps: f32,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
        weight_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [input_id, weight_id],
            saved_tensors: vec![input, weight],
            eps,
            input_grad_fns: [input_grad_fn, weight_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for RmsNormBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let saved_input = &self.saved_tensors[0];
        let saved_weight = &self.saved_tensors[1];
        let ndim = saved_input.ndim();
        let last_dim = ndim - 1;

        // Recompute rstd = 1 / sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
        let x_sq = client.mul(saved_input, saved_input)?;
        let mean_x_sq = client.mean(&x_sq, &[last_dim], true)?;
        let variance_eps = client.add_scalar(&mean_x_sq, self.eps as f64)?;
        let rms = client.sqrt(&variance_eps)?;
        let rstd = client.recip(&rms)?;

        // x_norm = x * rstd
        let x_norm = client.mul(saved_input, &rstd)?;

        // d_input = rstd * (grad_output * weight - x_norm * mean(grad_output * weight * x_norm))
        let gw = client.mul(grad_output, saved_weight)?;
        let gw_xn = client.mul(&gw, &x_norm)?;
        let mean_gw_xn = client.mean(&gw_xn, &[last_dim], true)?;
        let correction = client.mul(&x_norm, &mean_gw_xn)?;
        let inner = client.sub(&gw, &correction)?;
        let d_input = client.mul(&inner, &rstd)?;

        // d_weight = sum(grad_output * x_norm, batch_dims)
        let g_xn = client.mul(grad_output, &x_norm)?;
        let batch_dims: Vec<usize> = (0..last_dim).collect();
        let d_weight = if batch_dims.is_empty() {
            g_xn
        } else {
            client.sum(&g_xn, &batch_dims, false)?
        };

        Ok(vec![Some(d_input), Some(d_weight)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());
        let saved_input = &self.saved_tensors[0];
        let saved_weight = &self.saved_tensors[1];
        let ndim = saved_input.ndim();
        let last_dim = ndim - 1;

        // Recompute rstd and x_norm from saved tensors (treat as constants)
        let x_sq = client.mul(saved_input, saved_input)?;
        let mean_x_sq = client.mean(&x_sq, &[last_dim], true)?;
        let variance_eps = client.add_scalar(&mean_x_sq, self.eps as f64)?;
        let rms = client.sqrt(&variance_eps)?;
        let rstd = client.recip(&rms)?;
        let x_norm = client.mul(saved_input, &rstd)?;

        // Wrap as non-differentiable Vars (constants w.r.t. grad_output)
        let rstd_var = Var::new(rstd, false);
        let x_norm_var = Var::new(x_norm, false);
        let weight_var = Var::new(saved_weight.clone(), false);

        // d_input = rstd * (grad_output * weight - x_norm * mean(grad_output * weight * x_norm))
        let gw = var_mul(grad_output, &weight_var, &client)?;
        let gw_xn = var_mul(&gw, &x_norm_var, &client)?;
        let mean_gw_xn = var_mean(&gw_xn, &[last_dim], true, &client)?;
        let correction = var_mul(&x_norm_var, &mean_gw_xn, &client)?;
        let inner = var_sub(&gw, &correction, &client)?;
        let d_input = var_mul(&inner, &rstd_var, &client)?;

        // d_weight = sum(grad_output * x_norm, batch_dims)
        let g_xn = var_mul(grad_output, &x_norm_var, &client)?;
        let batch_dims: Vec<usize> = (0..last_dim).collect();
        let d_weight = if batch_dims.is_empty() {
            g_xn
        } else {
            var_sum(&g_xn, &batch_dims, false, &client)?
        };

        Ok(vec![Some(d_input), Some(d_weight)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        &self.saved_tensors
    }

    fn name(&self) -> &'static str {
        "RmsNormBackward"
    }
}

// ============================================================================
// LayerNormBackward
// ============================================================================

/// Backward for Layer Normalization: y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
///
/// Gradients:
/// - d_input = rstd * (gw - mean(gw) - x_norm * mean(gw * x_norm))
/// - d_weight = sum(grad_out * x_norm, batch_dims)
/// - d_bias = sum(grad_out, batch_dims)
///
/// Where gw = grad_out * weight, rstd = 1/sqrt(var+eps), x_norm = (x-mean)*rstd
pub struct LayerNormBackward<R: Runtime> {
    input_ids: [TensorId; 3],
    saved_tensors: Vec<Tensor<R>>, // [input, weight]
    eps: f32,
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 3],
}

impl<R: Runtime> LayerNormBackward<R> {
    /// Create a new LayerNormBackward
    pub fn new(
        input_id: TensorId,
        weight_id: TensorId,
        bias_id: TensorId,
        input: Tensor<R>,
        weight: Tensor<R>,
        eps: f32,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
        weight_grad_fn: Option<Arc<dyn GradFn<R>>>,
        bias_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [input_id, weight_id, bias_id],
            saved_tensors: vec![input, weight],
            eps,
            input_grad_fns: [input_grad_fn, weight_grad_fn, bias_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for LayerNormBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let saved_input = &self.saved_tensors[0];
        let saved_weight = &self.saved_tensors[1];
        let ndim = saved_input.ndim();
        let last_dim = ndim - 1;

        // Recompute rstd and x_norm
        // mean = mean(x, dim=-1, keepdim=True)
        let mu = client.mean(saved_input, &[last_dim], true)?;
        // x_centered = x - mean
        let x_centered = client.sub(saved_input, &mu)?;
        // var = mean(x_centered^2, dim=-1, keepdim=True)
        let x_centered_sq = client.mul(&x_centered, &x_centered)?;
        let variance = client.mean(&x_centered_sq, &[last_dim], true)?;
        // rstd = 1 / sqrt(var + eps)
        let variance_eps = client.add_scalar(&variance, self.eps as f64)?;
        let std = client.sqrt(&variance_eps)?;
        let rstd = client.recip(&std)?;
        // x_norm = x_centered * rstd
        let x_norm = client.mul(&x_centered, &rstd)?;

        // d_input = rstd * (gw - mean(gw) - x_norm * mean(gw * x_norm))
        let gw = client.mul(grad_output, saved_weight)?;
        let mean_gw = client.mean(&gw, &[last_dim], true)?;
        let gw_xn = client.mul(&gw, &x_norm)?;
        let mean_gw_xn = client.mean(&gw_xn, &[last_dim], true)?;
        let xn_mean_gw_xn = client.mul(&x_norm, &mean_gw_xn)?;
        let inner = client.sub(&gw, &mean_gw)?;
        let inner = client.sub(&inner, &xn_mean_gw_xn)?;
        let d_input = client.mul(&inner, &rstd)?;

        // d_weight = sum(grad_output * x_norm, batch_dims)
        let g_xn = client.mul(grad_output, &x_norm)?;
        let batch_dims: Vec<usize> = (0..last_dim).collect();
        let d_weight = if batch_dims.is_empty() {
            g_xn
        } else {
            client.sum(&g_xn, &batch_dims, false)?
        };

        // d_bias = sum(grad_output, batch_dims)
        let d_bias = if batch_dims.is_empty() {
            grad_output.clone()
        } else {
            client.sum(grad_output, &batch_dims, false)?
        };

        Ok(vec![Some(d_input), Some(d_weight), Some(d_bias)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());
        let saved_input = &self.saved_tensors[0];
        let saved_weight = &self.saved_tensors[1];
        let ndim = saved_input.ndim();
        let last_dim = ndim - 1;

        // Recompute from saved tensors (constants w.r.t. grad_output)
        let mu = client.mean(saved_input, &[last_dim], true)?;
        let x_centered = client.sub(saved_input, &mu)?;
        let x_centered_sq = client.mul(&x_centered, &x_centered)?;
        let variance = client.mean(&x_centered_sq, &[last_dim], true)?;
        let variance_eps = client.add_scalar(&variance, self.eps as f64)?;
        let std = client.sqrt(&variance_eps)?;
        let rstd = client.recip(&std)?;
        let x_norm = client.mul(&x_centered, &rstd)?;

        // Wrap as non-differentiable Vars
        let rstd_var = Var::new(rstd, false);
        let x_norm_var = Var::new(x_norm, false);
        let weight_var = Var::new(saved_weight.clone(), false);

        // d_input = rstd * (gw - mean(gw) - x_norm * mean(gw * x_norm))
        let gw = var_mul(grad_output, &weight_var, &client)?;
        let mean_gw = var_mean(&gw, &[last_dim], true, &client)?;
        let gw_xn = var_mul(&gw, &x_norm_var, &client)?;
        let mean_gw_xn = var_mean(&gw_xn, &[last_dim], true, &client)?;
        let xn_mean_gw_xn = var_mul(&x_norm_var, &mean_gw_xn, &client)?;
        let inner = var_sub(&gw, &mean_gw, &client)?;
        let inner = var_sub(&inner, &xn_mean_gw_xn, &client)?;
        let d_input = var_mul(&inner, &rstd_var, &client)?;

        // d_weight = sum(grad_output * x_norm, batch_dims)
        let g_xn = var_mul(grad_output, &x_norm_var, &client)?;
        let batch_dims: Vec<usize> = (0..last_dim).collect();
        let d_weight = if batch_dims.is_empty() {
            g_xn
        } else {
            var_sum(&g_xn, &batch_dims, false, &client)?
        };

        // d_bias = sum(grad_output, batch_dims)
        let d_bias = if batch_dims.is_empty() {
            grad_output.clone()
        } else {
            var_sum(grad_output, &batch_dims, false, &client)?
        };

        Ok(vec![Some(d_input), Some(d_weight), Some(d_bias)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        &self.saved_tensors
    }

    fn name(&self) -> &'static str {
        "LayerNormBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_rms_norm_backward_uniform() {
        let device = CpuDevice::new();

        // Input where all values are the same: rms_norm should just multiply by weight
        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 4], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device);
        let eps = 1e-5f32;

        let grad_out = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 0.0], &[1, 4], &device);

        let backward = RmsNormBackward::<CpuRuntime>::new(
            input.id(),
            weight.id(),
            input,
            weight,
            eps,
            None,
            None,
        );
        let grads = backward.backward(&grad_out).unwrap();

        assert_eq!(grads.len(), 2);
        let d_input: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        let d_weight: Vec<f32> = grads[1].as_ref().unwrap().to_vec();

        // With uniform input [1,1,1,1], rms = sqrt(1 + eps) ~ 1
        // x_norm ~ [1,1,1,1], grad_out*weight = [1,0,0,0]
        // mean(grad_out * weight * x_norm) ~ 0.25
        // d_input[0] ~ rstd * (1 - 1*0.25) = rstd * 0.75
        // d_input[1] ~ rstd * (0 - 1*0.25) = rstd * -0.25
        assert!(d_input[0] > 0.0, "d_input[0] should be positive");
        assert!(d_input[1] < 0.0, "d_input[1] should be negative");

        // d_weight should be sum(grad_out * x_norm, batch_dims)
        // grad_out * x_norm = [~1, 0, 0, 0]
        assert!((d_weight[0] - 1.0).abs() < 0.01);
        assert!(d_weight[1].abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_backward_gradient_sum() {
        // For RMS norm, the sum of d_input along the normalized dimension
        // should NOT be zero (unlike layer norm)
        let device = CpuDevice::new();

        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[1, 3], DType::F32, &device);

        let backward = RmsNormBackward::<CpuRuntime>::new(
            input.id(),
            weight.id(),
            input,
            weight,
            1e-5,
            None,
            None,
        );
        let grads = backward.backward(&grad_out).unwrap();
        let d_input: Vec<f32> = grads[0].as_ref().unwrap().to_vec();

        // Verify gradients are finite
        for val in &d_input {
            assert!(val.is_finite(), "gradient should be finite");
        }
    }

    #[test]
    fn test_layer_norm_backward_uniform_grad() {
        let device = CpuDevice::new();

        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device);
        let eps = 1e-5f32;

        // Uniform gradient
        let grad_out = Tensor::<CpuRuntime>::ones(&[1, 4], DType::F32, &device);

        let backward = LayerNormBackward::<CpuRuntime>::new(
            input.id(),
            weight.id(),
            TensorId::new(),
            input,
            weight,
            eps,
            None,
            None,
            None,
        );
        let grads = backward.backward(&grad_out).unwrap();

        assert_eq!(grads.len(), 3);
        let d_input: Vec<f32> = grads[0].as_ref().unwrap().to_vec();

        // For layer norm with uniform gradient and uniform weight,
        // d_input should be approximately zero (normalization removes mean)
        let sum: f32 = d_input.iter().sum();
        assert!(
            sum.abs() < 1e-5,
            "sum of d_input should be ~0 for uniform grad, got {}",
            sum
        );
    }

    #[test]
    fn test_layer_norm_backward_bias_grad() {
        let device = CpuDevice::new();

        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device);

        let grad_out = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);

        let backward = LayerNormBackward::<CpuRuntime>::new(
            input.id(),
            weight.id(),
            TensorId::new(),
            input,
            weight,
            1e-5,
            None,
            None,
            None,
        );
        let grads = backward.backward(&grad_out).unwrap();

        let d_bias: Vec<f32> = grads[2].as_ref().unwrap().to_vec();

        // d_bias = sum(grad_output, batch_dims) = sum along dim 0
        // d_bias[0] = 1.0 + 3.0 = 4.0
        // d_bias[1] = 2.0 + 4.0 = 6.0
        assert!((d_bias[0] - 4.0).abs() < 1e-5);
        assert!((d_bias[1] - 6.0).abs() < 1e-5);
    }
}
