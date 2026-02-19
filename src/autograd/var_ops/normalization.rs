//! Normalization operations (rms_norm, layer_norm)

use super::ops::*;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::{NormalizationOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// RMS Normalization: y = x / rms(x) * weight
///
/// Uses the fused `NormalizationOps::rms_norm` kernel for the forward pass
/// and tracks gradients for both input and weight.
///
/// # Arguments
///
/// * `input` - Input variable of shape `[..., hidden_size]`
/// * `weight` - Weight variable of shape `[hidden_size]`
/// * `eps` - Small constant for numerical stability
/// * `client` - Runtime client
pub fn var_rms_norm<R, C>(input: &Var<R>, weight: &Var<R>, eps: f32, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + NormalizationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.rms_norm(input.tensor(), weight.tensor(), eps)?;

    if input.requires_grad() || weight.requires_grad() {
        let grad_fn = RmsNormBackward::<R>::new(
            input.id(),
            weight.id(),
            input.tensor().clone(),
            weight.tensor().clone(),
            eps,
            input.grad_fn().cloned(),
            weight.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Layer Normalization: y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
///
/// Uses the fused `NormalizationOps::layer_norm` kernel for the forward pass
/// and tracks gradients for input, weight, and bias.
///
/// # Arguments
///
/// * `input` - Input variable of shape `[..., hidden_size]`
/// * `weight` - Weight (gamma) variable of shape `[hidden_size]`
/// * `bias` - Bias (beta) variable of shape `[hidden_size]`
/// * `eps` - Small constant for numerical stability
/// * `client` - Runtime client
pub fn var_layer_norm<R, C>(
    input: &Var<R>,
    weight: &Var<R>,
    bias: &Var<R>,
    eps: f32,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + NormalizationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.layer_norm(input.tensor(), weight.tensor(), bias.tensor(), eps)?;

    if input.requires_grad() || weight.requires_grad() || bias.requires_grad() {
        let grad_fn = LayerNormBackward::<R>::new(
            input.id(),
            weight.id(),
            bias.id(),
            input.tensor().clone(),
            weight.tensor().clone(),
            eps,
            input.grad_fn().cloned(),
            weight.grad_fn().cloned(),
            bias.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_var_rms_norm_forward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device),
            true,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device),
            true,
        );

        let result = var_rms_norm(&input, &weight, 1e-5, &client).unwrap();
        let data: Vec<f32> = result.tensor().to_vec();

        // rms = sqrt(mean([1, 4, 9, 16]) + 1e-5) = sqrt(7.5 + 1e-5) ~ 2.7386
        // output = [1/rms, 2/rms, 3/rms, 4/rms] * [1,1,1,1]
        let rms = (7.5f32 + 1e-5).sqrt();
        for i in 0..4 {
            let expected = (i as f32 + 1.0) / rms;
            assert!(
                (data[i] - expected).abs() < 1e-5,
                "data[{}] = {}, expected {}",
                i,
                data[i],
                expected,
            );
        }
    }

    #[test]
    fn test_var_rms_norm_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device),
            true,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device),
            true,
        );

        let output = var_rms_norm(&input, &weight, 1e-5, &client).unwrap();

        // Sum the output to get a scalar for backward
        // Sum over all dims to get a scalar for backward
        let loss = crate::autograd::var_sum(&output, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let grad_input = grads.get(input.id()).unwrap();
        let grad_weight = grads.get(weight.id()).unwrap();

        let gi: Vec<f32> = grad_input.to_vec();
        let gw: Vec<f32> = grad_weight.to_vec();

        // Verify gradients are finite and have correct shapes
        assert_eq!(gi.len(), 3);
        assert_eq!(gw.len(), 3);
        for val in &gi {
            assert!(val.is_finite(), "input gradient should be finite");
        }
        for val in &gw {
            assert!(val.is_finite(), "weight gradient should be finite");
        }
    }

    #[test]
    fn test_var_layer_norm_forward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device),
            true,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device),
            true,
        );
        let bias = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device),
            true,
        );

        let result = var_layer_norm(&input, &weight, &bias, 1e-5, &client).unwrap();
        let data: Vec<f32> = result.tensor().to_vec();

        // mean = 2.5, var = mean([(-1.5)^2, (-0.5)^2, (0.5)^2, (1.5)^2]) = 1.25
        // rstd = 1/sqrt(1.25 + 1e-5)
        // output should have mean ~0 and unit variance
        let sum: f32 = data.iter().sum();
        assert!(sum.abs() < 1e-4, "layer norm output should have ~0 mean");
    }

    #[test]
    fn test_var_layer_norm_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device),
            true,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device),
            true,
        );
        let bias = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0], &[3], &device),
            true,
        );

        let output = var_layer_norm(&input, &weight, &bias, 1e-5, &client).unwrap();

        // Sum over all dims to get a scalar for backward
        let loss = crate::autograd::var_sum(&output, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let grad_input = grads.get(input.id()).unwrap();
        let grad_weight = grads.get(weight.id()).unwrap();
        let grad_bias = grads.get(bias.id()).unwrap();

        let gi: Vec<f32> = grad_input.to_vec();
        let gw: Vec<f32> = grad_weight.to_vec();
        let gb: Vec<f32> = grad_bias.to_vec();

        // Verify shapes
        assert_eq!(gi.len(), 3);
        assert_eq!(gw.len(), 3);
        assert_eq!(gb.len(), 3);

        // For layer norm with sum loss:
        // d_bias = sum(grad_output) = [1, 1, 1] (each element contributes 1)
        for val in &gb {
            assert!(
                (*val - 1.0).abs() < 1e-5,
                "bias gradient should be 1.0, got {}",
                val,
            );
        }

        // d_input should sum to ~0 (layer norm property)
        let sum: f32 = gi.iter().sum();
        assert!(
            sum.abs() < 1e-5,
            "sum of input gradients should be ~0, got {}",
            sum,
        );

        // All gradients should be finite
        for val in &gi {
            assert!(val.is_finite());
        }
        for val in &gw {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_var_rms_norm_no_grad() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // When no inputs require grad, output should not track gradients
        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device),
            false,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device),
            false,
        );

        let result = var_rms_norm(&input, &weight, 1e-5, &client).unwrap();
        assert!(!result.requires_grad());
        assert!(result.grad_fn().is_none());
    }
}
