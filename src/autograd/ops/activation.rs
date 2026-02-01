//! Backward implementations for activation functions
//!
//! Implements gradient computation for relu, sigmoid, and softmax.

use crate::autograd::GradFn;
use crate::error::Result;
use crate::ops::{BinaryOps, CompareOps, ReduceOps, TensorOps};
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};

#[cfg(test)]
use crate::ops::ActivationOps;

// ============================================================================
// ReluBackward
// ============================================================================

/// Backward for ReLU: z = max(0, a)
///
/// Gradient: dL/da = dL/dz * (a > 0)
/// The gradient passes through where input was positive, zero elsewhere.
pub struct ReluBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
}

impl<R: Runtime> ReluBackward<R> {
    /// Create a new ReluBackward
    pub fn new(input_id: TensorId, input: Tensor<R>) -> Self {
        Self {
            input_id,
            saved_input: input,
        }
    }
}

impl<R: Runtime> GradFn<R> for ReluBackward<R>
where
    R::Client: TensorOps<R> + CompareOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());

        // ReLU derivative: relu'(x) = 1 if x > 0, 0 otherwise
        // mask = (input > 0) -> returns 1.0 where true, 0.0 where false
        let zero = Tensor::<R>::zeros(
            self.saved_input.shape(),
            self.saved_input.dtype(),
            self.saved_input.device(),
        );
        let mask = client.gt(&self.saved_input, &zero)?;

        // grad = grad_output * mask
        let grad = client.mul(grad_output, &mask)?;

        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_input)
    }

    fn name(&self) -> &'static str {
        "ReluBackward"
    }
}

// ============================================================================
// SigmoidBackward
// ============================================================================

/// Backward for sigmoid: z = 1 / (1 + exp(-a))
///
/// Gradient: dL/da = dL/dz * z * (1 - z)
pub struct SigmoidBackward<R: Runtime> {
    input_id: TensorId,
    saved_output: Tensor<R>, // sigmoid(a)
}

impl<R: Runtime> SigmoidBackward<R> {
    /// Create a new SigmoidBackward
    pub fn new(input_id: TensorId, output: Tensor<R>) -> Self {
        Self {
            input_id,
            saved_output: output,
        }
    }
}

impl<R: Runtime> GradFn<R> for SigmoidBackward<R>
where
    R::Client: TensorOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());

        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        // dL/da = dL/dz * z * (1 - z)
        let one = Tensor::<R>::ones(
            self.saved_output.shape(),
            self.saved_output.dtype(),
            self.saved_output.device(),
        );
        let one_minus_sigmoid = client.sub(&one, &self.saved_output)?;
        let sigmoid_deriv = client.mul(&self.saved_output, &one_minus_sigmoid)?;
        let grad = client.mul(grad_output, &sigmoid_deriv)?;

        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_output)
    }

    fn name(&self) -> &'static str {
        "SigmoidBackward"
    }
}

// ============================================================================
// SoftmaxBackward
// ============================================================================

/// Backward for softmax along a dimension: z_i = exp(a_i) / sum(exp(a))
///
/// Gradient: dL/da_i = z_i * (dL/dz_i - sum_j(z_j * dL/dz_j))
/// Simplified: dL/da = z * (dL/dz - sum(z * dL/dz, dim=dim))
pub struct SoftmaxBackward<R: Runtime> {
    input_id: TensorId,
    saved_output: Tensor<R>, // softmax(a)
    dim: isize,
}

impl<R: Runtime> SoftmaxBackward<R> {
    /// Create a new SoftmaxBackward
    pub fn new(input_id: TensorId, output: Tensor<R>, dim: isize) -> Self {
        Self {
            input_id,
            saved_output: output,
            dim,
        }
    }
}

impl<R: Runtime> GradFn<R> for SoftmaxBackward<R>
where
    R::Client: TensorOps<R> + ReduceOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());

        // Normalize dim
        let ndim = self.saved_output.ndim();
        let dim_idx = if self.dim < 0 {
            (ndim as isize + self.dim) as usize
        } else {
            self.dim as usize
        };

        // softmax gradient: grad_input = softmax * (grad_output - sum(softmax * grad_output, dim))
        // = z * (dy - sum(z * dy, dim))
        let z_dy = client.mul(&self.saved_output, grad_output)?;
        let sum_z_dy = client.sum(&z_dy, &[dim_idx], true)?;
        let dy_minus_sum = client.sub(grad_output, &sum_z_dy)?;
        let grad = client.mul(&self.saved_output, &dy_minus_sum)?;

        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_output)
    }

    fn name(&self) -> &'static str {
        "SoftmaxBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_relu_backward_positive() {
        let device = CpuDevice::new();

        // Input with positive values
        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[3], DType::F32, &device);

        let backward = ReluBackward::<CpuRuntime>::new(input.id(), input);
        let grads = backward.backward(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        // All positive inputs, so gradient passes through
        for val in grad_data {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_relu_backward_mixed() {
        let device = CpuDevice::new();

        // Input with mixed values
        let input = Tensor::<CpuRuntime>::from_slice(&[-1.0f32, 0.0, 2.0], &[3], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[3], DType::F32, &device);

        let backward = ReluBackward::<CpuRuntime>::new(input.id(), input);
        let grads = backward.backward(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        // grad[0] should be ~0 (negative input)
        // grad[1] should be ~0 (zero input)
        // grad[2] should be ~1 (positive input)
        assert!(grad_data[0].abs() < 1e-5);
        assert!(grad_data[1].abs() < 1e-5);
        assert!((grad_data[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // sigmoid(0) = 0.5
        // sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25
        let input = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device);
        let output = client.sigmoid(&input).unwrap();

        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device);

        let backward = SigmoidBackward::<CpuRuntime>::new(input.id(), output);
        let grads = backward.backward(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert!((grad_data[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Simple 2-element softmax
        let input = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device);
        let output = client.softmax(&input, -1).unwrap(); // [0.5, 0.5]

        // dL/dz = [1, 0] - only first element contributes to loss
        let grad_out = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0], &[2], &device);

        let backward = SoftmaxBackward::<CpuRuntime>::new(input.id(), output, -1);
        let grads = backward.backward(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        // For softmax, when z = [0.5, 0.5] and dy = [1, 0]:
        // grad = z * (dy - sum(z * dy))
        // sum(z * dy) = 0.5 * 1 + 0.5 * 0 = 0.5
        // grad[0] = 0.5 * (1 - 0.5) = 0.25
        // grad[1] = 0.5 * (0 - 0.5) = -0.25
        assert!((grad_data[0] - 0.25).abs() < 1e-6);
        assert!((grad_data[1] - (-0.25)).abs() < 1e-6);
    }
}
