//! Backward implementations for reduction operations
//!
//! Implements gradient computation for sum and mean reductions.

use crate::autograd::GradFn;
use crate::error::Result;
use crate::ops::ScalarOps;
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use std::marker::PhantomData;

// ============================================================================
// SumBackward
// ============================================================================

/// Backward for sum reduction: z = sum(a, dims)
///
/// The gradient of sum is broadcast expansion.
/// For z = sum(a, dims), dL/da = broadcast(dL/dz, original_shape)
///
/// If keepdim=false, we need to unsqueeze the gradient before broadcasting.
pub struct SumBackward<R: Runtime> {
    input_id: TensorId,
    input_shape: Vec<usize>,
    dims: Vec<usize>,
    keepdim: bool,
    _marker: PhantomData<R>,
}

impl<R: Runtime> SumBackward<R> {
    /// Create a new SumBackward
    pub fn new(input_id: TensorId, input_shape: &[usize], dims: &[usize], keepdim: bool) -> Self {
        Self {
            input_id,
            input_shape: input_shape.to_vec(),
            dims: dims.to_vec(),
            keepdim,
            _marker: PhantomData,
        }
    }
}

impl<R: Runtime> GradFn<R> for SumBackward<R> {
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        // For sum, the gradient is broadcast back to the original shape
        // All elements contribute equally to the sum, so each gets the full gradient

        let mut grad = grad_output.clone();

        // If keepdim=false, we need to unsqueeze the dimensions that were reduced
        if !self.keepdim {
            // Sort dims in ascending order to unsqueeze correctly
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();

            for &dim in &sorted_dims {
                grad = grad.unsqueeze(dim as isize)?;
            }
        }

        // Broadcast to original shape
        grad = grad.broadcast_to(&self.input_shape)?;

        // Make contiguous if needed
        if !grad.is_contiguous() {
            grad = grad.contiguous();
        }

        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}

// ============================================================================
// MeanBackward
// ============================================================================

/// Backward for mean reduction: z = mean(a, dims)
///
/// For z = mean(a, dims), dL/da = broadcast(dL/dz, original_shape) / count
/// where count is the number of elements being averaged.
pub struct MeanBackward<R: Runtime> {
    input_id: TensorId,
    input_shape: Vec<usize>,
    dims: Vec<usize>,
    keepdim: bool,
    _marker: PhantomData<R>,
}

impl<R: Runtime> MeanBackward<R> {
    /// Create a new MeanBackward
    pub fn new(input_id: TensorId, input_shape: &[usize], dims: &[usize], keepdim: bool) -> Self {
        Self {
            input_id,
            input_shape: input_shape.to_vec(),
            dims: dims.to_vec(),
            keepdim,
            _marker: PhantomData,
        }
    }
}

impl<R: Runtime> GradFn<R> for MeanBackward<R>
where
    R::Client: ScalarOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());

        // Calculate the count (number of elements being averaged)
        let count: usize = self.dims.iter().map(|&d| self.input_shape[d]).product();
        let count_f64 = count as f64;

        let mut grad = grad_output.clone();

        // If keepdim=false, we need to unsqueeze the dimensions that were reduced
        if !self.keepdim {
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();

            for &dim in &sorted_dims {
                grad = grad.unsqueeze(dim as isize)?;
            }
        }

        // Broadcast to original shape
        grad = grad.broadcast_to(&self.input_shape)?;

        // Make contiguous if needed
        if !grad.is_contiguous() {
            grad = grad.contiguous();
        }

        // Divide by count
        let grad = client.div_scalar(&grad, count_f64)?;

        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_sum_backward_keepdim() {
        let device = CpuDevice::new();

        // a = [[1, 2, 3], [4, 5, 6]] (2x3)
        // sum(a, dim=1, keepdim=True) = [[6], [15]] (2x1)
        // dL/dz = [[1], [1]] (2x1)
        // dL/da = [[1, 1, 1], [1, 1, 1]] (2x3)

        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 1], DType::F32, &device);

        let backward = SumBackward::<CpuRuntime>::new(
            TensorId::new(),
            &[2, 3],
            &[1],
            true, // keepdim
        );
        let grads = backward.backward(&grad_out).unwrap();

        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 3]);

        let grad_data: Vec<f32> = grad_a.to_vec();
        assert_eq!(grad_data, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sum_backward_no_keepdim() {
        let device = CpuDevice::new();

        // a = [[1, 2, 3], [4, 5, 6]] (2x3)
        // sum(a, dim=1, keepdim=False) = [6, 15] (2,)
        // dL/dz = [1, 1] (2,)
        // dL/da = [[1, 1, 1], [1, 1, 1]] (2x3)

        let grad_out = Tensor::<CpuRuntime>::ones(&[2], DType::F32, &device);

        let backward = SumBackward::<CpuRuntime>::new(
            TensorId::new(),
            &[2, 3],
            &[1],
            false, // no keepdim
        );
        let grads = backward.backward(&grad_out).unwrap();

        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 3]);

        let grad_data: Vec<f32> = grad_a.to_vec();
        assert_eq!(grad_data, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mean_backward() {
        let device = CpuDevice::new();

        // a = [[1, 2, 3], [4, 5, 6]] (2x3)
        // mean(a, dim=1, keepdim=True) = [[2], [5]] (2x1)
        // dL/dz = [[1], [1]] (2x1)
        // dL/da = [[1/3, 1/3, 1/3], [1/3, 1/3, 1/3]] (2x3)

        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 1], DType::F32, &device);

        let backward = MeanBackward::<CpuRuntime>::new(
            TensorId::new(),
            &[2, 3],
            &[1],
            true, // keepdim
        );
        let grads = backward.backward(&grad_out).unwrap();

        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 3]);

        let grad_data: Vec<f32> = grad_a.to_vec();
        let expected = 1.0 / 3.0;
        for val in grad_data {
            assert!((val - expected).abs() < 1e-6);
        }
    }
}
