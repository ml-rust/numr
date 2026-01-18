//! Backward implementations for reduction operations
//!
//! Implements gradient computation for sum, mean, max, and min reductions.

use crate::autograd::GradFn;
use crate::error::Result;
use crate::ops::{CompareOps, ScalarOps, TensorOps};
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

// ============================================================================
// Helper Functions
// ============================================================================

/// Ensure a tensor is contiguous, making a copy if necessary.
#[inline]
fn ensure_contiguous<R: Runtime>(tensor: Tensor<R>) -> Tensor<R> {
    if tensor.is_contiguous() {
        tensor
    } else {
        tensor.contiguous()
    }
}

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
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> SumBackward<R> {
    /// Create a new SumBackward
    pub fn new(
        input_id: TensorId,
        input_shape: &[usize],
        dims: &[usize],
        keepdim: bool,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            input_shape: input_shape.to_vec(),
            dims: dims.to_vec(),
            keepdim,
            input_grad_fn,
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

        // Broadcast to original shape and ensure contiguous
        grad = ensure_contiguous(grad.broadcast_to(&self.input_shape)?);

        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
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
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> MeanBackward<R> {
    /// Create a new MeanBackward
    pub fn new(
        input_id: TensorId,
        input_shape: &[usize],
        dims: &[usize],
        keepdim: bool,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            input_shape: input_shape.to_vec(),
            dims: dims.to_vec(),
            keepdim,
            input_grad_fn,
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

        // Broadcast to original shape and ensure contiguous
        grad = ensure_contiguous(grad.broadcast_to(&self.input_shape)?);

        // Divide by count
        let grad = client.div_scalar(&grad, count_f64)?;

        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}

// ============================================================================
// MaxBackward
// ============================================================================

/// Backward for max reduction: z = max(a, dims)
///
/// The gradient flows only to the element(s) that had the maximum value.
/// For ties, the gradient is distributed equally among tied elements.
pub struct MaxBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    dims: Vec<usize>,
    keepdim: bool,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> MaxBackward<R> {
    /// Create a new MaxBackward
    pub fn new(
        input_id: TensorId,
        input: Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_input: input,
            dims: dims.to_vec(),
            keepdim,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for MaxBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());

        // Recompute max to get the max values
        let max_vals = client.max(&self.saved_input, &self.dims, true)?;

        // Broadcast max values to input shape for comparison
        let max_broadcast = ensure_contiguous(max_vals.broadcast_to(self.saved_input.shape())?);

        // Create mask where input equals max (handles ties)
        let mask = client.eq(&self.saved_input, &max_broadcast)?;

        // Count how many elements equal the max per reduction group (for distributing gradient in case of ties)
        let mask_sum = client.sum(&mask, &self.dims, true)?;

        // Broadcast mask_sum to input shape
        let mask_sum_broadcast =
            ensure_contiguous(mask_sum.broadcast_to(self.saved_input.shape())?);

        // Normalize mask by count (distribute gradient equally among tied elements)
        let normalized_mask = client.div(&mask, &mask_sum_broadcast)?;

        // Broadcast grad_output to input shape
        let mut grad = grad_output.clone();
        if !self.keepdim {
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();
            for &dim in &sorted_dims {
                grad = grad.unsqueeze(dim as isize)?;
            }
        }
        let grad_broadcast = ensure_contiguous(grad.broadcast_to(self.saved_input.shape())?);

        // Multiply gradient by normalized mask
        let grad_input = client.mul(&grad_broadcast, &normalized_mask)?;

        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_input)
    }

    fn name(&self) -> &'static str {
        "MaxBackward"
    }
}

// ============================================================================
// MinBackward
// ============================================================================

/// Backward for min reduction: z = min(a, dims)
///
/// The gradient flows only to the element(s) that had the minimum value.
/// For ties, the gradient is distributed equally among tied elements.
pub struct MinBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    dims: Vec<usize>,
    keepdim: bool,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> MinBackward<R> {
    /// Create a new MinBackward
    pub fn new(
        input_id: TensorId,
        input: Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_input: input,
            dims: dims.to_vec(),
            keepdim,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for MinBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());

        // Recompute min to get the min values
        let min_vals = client.min(&self.saved_input, &self.dims, true)?;

        // Broadcast min values to input shape for comparison
        let min_broadcast = ensure_contiguous(min_vals.broadcast_to(self.saved_input.shape())?);

        // Create mask where input equals min (handles ties)
        let mask = client.eq(&self.saved_input, &min_broadcast)?;

        // Count how many elements equal the min per reduction group
        let mask_sum = client.sum(&mask, &self.dims, true)?;

        // Broadcast mask_sum to input shape
        let mask_sum_broadcast =
            ensure_contiguous(mask_sum.broadcast_to(self.saved_input.shape())?);

        // Normalize mask by count
        let normalized_mask = client.div(&mask, &mask_sum_broadcast)?;

        // Broadcast grad_output to input shape
        let mut grad = grad_output.clone();
        if !self.keepdim {
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();
            for &dim in &sorted_dims {
                grad = grad.unsqueeze(dim as isize)?;
            }
        }
        let grad_broadcast = ensure_contiguous(grad.broadcast_to(self.saved_input.shape())?);

        // Multiply gradient by normalized mask
        let grad_input = client.mul(&grad_broadcast, &normalized_mask)?;

        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_input)
    }

    fn name(&self) -> &'static str {
        "MinBackward"
    }
}

// ============================================================================
// VarBackward
// ============================================================================

/// Backward for variance reduction: z = var(a, dims, correction)
///
/// The gradient of variance is:
/// dL/da = dL/dz * 2 * (a - mean(a)) / (N - correction)
///
/// where N is the number of elements being reduced.
pub struct VarBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    dims: Vec<usize>,
    keepdim: bool,
    correction: usize,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> VarBackward<R> {
    /// Create a new VarBackward
    pub fn new(
        input_id: TensorId,
        input: Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_input: input,
            dims: dims.to_vec(),
            keepdim,
            correction,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for VarBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());

        // Calculate N (number of elements in reduction)
        let n: usize = self
            .dims
            .iter()
            .map(|&d| self.saved_input.shape()[d])
            .product();
        let n_minus_corr = (n - self.correction) as f64;

        // Compute mean of input
        let mean = client.mean(&self.saved_input, &self.dims, true)?;

        // Broadcast mean to input shape
        let mean_broadcast = ensure_contiguous(mean.broadcast_to(self.saved_input.shape())?);

        // a - mean(a)
        let centered = client.sub(&self.saved_input, &mean_broadcast)?;

        // 2 * (a - mean) / (N - correction)
        let scale = 2.0 / n_minus_corr;
        let grad_contrib = client.mul_scalar(&centered, scale)?;

        // Handle grad_output shape - broadcast to input shape
        let mut grad = grad_output.clone();
        if !self.keepdim {
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();
            for &dim in &sorted_dims {
                grad = grad.unsqueeze(dim as isize)?;
            }
        }
        let grad_broadcast = ensure_contiguous(grad.broadcast_to(self.saved_input.shape())?);

        // Final gradient
        let grad_input = client.mul(&grad_broadcast, &grad_contrib)?;

        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_input)
    }

    fn name(&self) -> &'static str {
        "VarBackward"
    }
}

// ============================================================================
// StdBackward
// ============================================================================

/// Backward for standard deviation reduction: z = std(a, dims, correction)
///
/// std = sqrt(var), so by chain rule:
/// dL/da = dL/dz * d(sqrt(var))/dvar * dvar/da
///       = dL/dz * 1/(2*std) * 2*(a - mean) / (N - correction)
///       = dL/dz * (a - mean) / ((N - correction) * std)
pub struct StdBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    saved_output: Tensor<R>, // std(a)
    dims: Vec<usize>,
    keepdim: bool,
    correction: usize,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> StdBackward<R> {
    /// Create a new StdBackward
    pub fn new(
        input_id: TensorId,
        input: Tensor<R>,
        output: Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_input: input,
            saved_output: output,
            dims: dims.to_vec(),
            keepdim,
            correction,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for StdBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());

        // Calculate N (number of elements in reduction)
        let n: usize = self
            .dims
            .iter()
            .map(|&d| self.saved_input.shape()[d])
            .product();
        let n_minus_corr = (n - self.correction) as f64;

        // Compute mean of input
        let mean = client.mean(&self.saved_input, &self.dims, true)?;

        // Broadcast mean and std to input shape
        let mean_broadcast = ensure_contiguous(mean.broadcast_to(self.saved_input.shape())?);

        let std_for_broadcast = if self.keepdim {
            self.saved_output.clone()
        } else {
            let mut std_expanded = self.saved_output.clone();
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();
            for &dim in &sorted_dims {
                std_expanded = std_expanded.unsqueeze(dim as isize)?;
            }
            std_expanded
        };
        let std_broadcast =
            ensure_contiguous(std_for_broadcast.broadcast_to(self.saved_input.shape())?);

        // (a - mean)
        let centered = client.sub(&self.saved_input, &mean_broadcast)?;

        // (a - mean) / ((N - correction) * std)
        let denominator = client.mul_scalar(&std_broadcast, n_minus_corr)?;
        let grad_contrib = client.div(&centered, &denominator)?;

        // Handle grad_output shape - broadcast to input shape
        let mut grad = grad_output.clone();
        if !self.keepdim {
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();
            for &dim in &sorted_dims {
                grad = grad.unsqueeze(dim as isize)?;
            }
        }
        let grad_broadcast = ensure_contiguous(grad.broadcast_to(self.saved_input.shape())?);

        // Final gradient
        let grad_input = client.mul(&grad_broadcast, &grad_contrib)?;

        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        // Return both saved tensors - but we can only return a slice, so just input for now
        std::slice::from_ref(&self.saved_input)
    }

    fn name(&self) -> &'static str {
        "StdBackward"
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
            None, // input_grad_fn
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
            None,  // input_grad_fn
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
            None, // input_grad_fn
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

    #[test]
    fn test_max_backward() {
        let device = CpuDevice::new();
        let _client = CpuRuntime::default_client(&device);

        // a = [[1, 3, 2], [4, 2, 5]] (2x3)
        // max(a, dim=1, keepdim=True) = [[3], [5]] (2x1)
        // dL/dz = [[1], [1]] (2x1)
        // dL/da = [[0, 1, 0], [0, 0, 1]] (gradient flows only to max elements)
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0], &[2, 3], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 1], DType::F32, &device);

        let backward = MaxBackward::<CpuRuntime>::new(a.id(), a.clone(), &[1], true, None);
        let grads = backward.backward(&grad_out).unwrap();

        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 3]);

        let grad_data: Vec<f32> = grad_a.to_vec();
        // Max at index 1 for first row, index 2 for second row
        assert_eq!(grad_data, vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_min_backward() {
        let device = CpuDevice::new();
        let _client = CpuRuntime::default_client(&device);

        // a = [[3, 1, 2], [4, 2, 5]] (2x3)
        // min(a, dim=1, keepdim=True) = [[1], [2]] (2x1)
        // dL/dz = [[1], [1]] (2x1)
        // dL/da = [[0, 1, 0], [0, 1, 0]] (gradient flows only to min elements)
        let a =
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 2.0, 4.0, 2.0, 5.0], &[2, 3], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 1], DType::F32, &device);

        let backward = MinBackward::<CpuRuntime>::new(a.id(), a.clone(), &[1], true, None);
        let grads = backward.backward(&grad_out).unwrap();

        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 3]);

        let grad_data: Vec<f32> = grad_a.to_vec();
        // Min at index 1 for first row, index 1 for second row
        assert_eq!(grad_data, vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_max_backward_with_ties() {
        let device = CpuDevice::new();
        let _client = CpuRuntime::default_client(&device);

        // a = [[3, 3, 1]] (1x3) - two tied max values
        // max(a, dim=1, keepdim=True) = [[3]] (1x1)
        // dL/dz = [[1]] (1x1)
        // dL/da = [[0.5, 0.5, 0]] (gradient split equally among tied max elements)
        let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 3.0, 1.0], &[1, 3], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[1, 1], DType::F32, &device);

        let backward = MaxBackward::<CpuRuntime>::new(a.id(), a.clone(), &[1], true, None);
        let grads = backward.backward(&grad_out).unwrap();

        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[1, 3]);

        let grad_data: Vec<f32> = grad_a.to_vec();
        // Gradient split equally among two max elements
        assert!((grad_data[0] - 0.5).abs() < 1e-6);
        assert!((grad_data[1] - 0.5).abs() < 1e-6);
        assert!((grad_data[2] - 0.0).abs() < 1e-6);
    }
}
