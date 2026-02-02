//! Backward implementations for shape operations
//!
//! These implement gradient computation for shape-changing operations
//! like reshape, transpose, squeeze, and unsqueeze.
//!
//! # Key Property
//!
//! Shape operations have identity Jacobians - the gradient of reshaping
//! is just reshaping the gradient back to the original shape.

use crate::autograd::{GradFn, Var};
use crate::error::Result;
use crate::ops::ReduceOps;
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

// ============================================================================
// ReshapeBackward
// ============================================================================

/// Backward for reshape: z = reshape(a, new_shape)
///
/// Gradient: dL/da = reshape(dL/dz, original_shape)
/// The gradient simply reshapes back to the input's original shape.
pub struct ReshapeBackward<R: Runtime> {
    input_id: TensorId,
    input_shape: Vec<usize>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> ReshapeBackward<R> {
    /// Create a new ReshapeBackward
    pub fn new(
        input_id: TensorId,
        input_shape: Vec<usize>,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            input_shape,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for ReshapeBackward<R> {
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        // Reshape gradient back to input shape
        let grad = grad_output.reshape(&self.input_shape)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        // Reshape gradient back to input shape
        // Reshape is a view operation with identity Jacobian, so we just
        // reshape and preserve the gradient function from grad_output
        let reshaped = grad_output.tensor().reshape(&self.input_shape)?;

        // Create ReshapeBackward to track the chain
        let grad_fn = ReshapeBackward::<R>::new(
            grad_output.id(),
            grad_output.shape().to_vec(),
            grad_output.grad_fn().cloned(),
        );

        if grad_output.requires_grad() {
            Ok(vec![Some(Var::from_op(reshaped, Arc::new(grad_fn)))])
        } else {
            Ok(vec![Some(Var::new(reshaped, false))])
        }
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "ReshapeBackward"
    }
}

// ============================================================================
// TransposeBackward
// ============================================================================

/// Backward for transpose: z = transpose(a)
///
/// Gradient: dL/da = transpose(dL/dz)
/// The gradient is just the transpose of the output gradient.
pub struct TransposeBackward<R: Runtime> {
    input_id: TensorId,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> TransposeBackward<R> {
    /// Create a new TransposeBackward
    pub fn new(input_id: TensorId, input_grad_fn: Option<Arc<dyn GradFn<R>>>) -> Self {
        Self {
            input_id,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for TransposeBackward<R> {
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        // Transpose gradient back
        let grad = grad_output.t()?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        // Transpose gradient back
        let transposed = grad_output.tensor().t()?;

        // Create TransposeBackward to track the chain
        let grad_fn = TransposeBackward::<R>::new(grad_output.id(), grad_output.grad_fn().cloned());

        if grad_output.requires_grad() {
            Ok(vec![Some(Var::from_op(transposed, Arc::new(grad_fn)))])
        } else {
            Ok(vec![Some(Var::new(transposed, false))])
        }
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "TransposeBackward"
    }
}

// ============================================================================
// PermuteBackward
// ============================================================================

/// Backward for permute: z = permute(a, dims)
///
/// Gradient: dL/da = permute(dL/dz, inverse_dims)
/// The gradient uses the inverse permutation.
pub struct PermuteBackward<R: Runtime> {
    input_id: TensorId,
    inverse_dims: Vec<usize>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> PermuteBackward<R> {
    /// Create a new PermuteBackward
    ///
    /// # Arguments
    /// * `input_id` - ID of the input tensor
    /// * `dims` - The permutation dimensions used in forward pass
    /// * `input_grad_fn` - Gradient function of input
    pub fn new(
        input_id: TensorId,
        dims: &[usize],
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        // Compute inverse permutation
        let mut inverse_dims = vec![0; dims.len()];
        for (i, &d) in dims.iter().enumerate() {
            inverse_dims[d] = i;
        }

        Self {
            input_id,
            inverse_dims,
            input_grad_fn,
        }
    }

    /// Create from inverse dims directly (for backward_var chaining)
    fn from_inverse(
        input_id: TensorId,
        inverse_dims: Vec<usize>,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            inverse_dims,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for PermuteBackward<R> {
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        // Apply inverse permutation
        let grad = grad_output.permute(&self.inverse_dims)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        // Apply inverse permutation
        let permuted = grad_output.tensor().permute(&self.inverse_dims)?;

        // The inverse of the inverse is the original permutation
        // So we pass self.inverse_dims as the "forward" dims for the new backward
        let grad_fn = PermuteBackward::<R>::from_inverse(
            grad_output.id(),
            self.inverse_dims.clone(),
            grad_output.grad_fn().cloned(),
        );

        if grad_output.requires_grad() {
            Ok(vec![Some(Var::from_op(permuted, Arc::new(grad_fn)))])
        } else {
            Ok(vec![Some(Var::new(permuted, false))])
        }
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "PermuteBackward"
    }
}

// ============================================================================
// ExpandBackward (for broadcast_to)
// ============================================================================

/// Backward for expand/broadcast_to: z = broadcast_to(a, new_shape)
///
/// Gradient: dL/da = sum(dL/dz, broadcast_dims)
/// The gradient sums over dimensions that were broadcast.
pub struct ExpandBackward<R: Runtime> {
    input_id: TensorId,
    input_shape: Vec<usize>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> ExpandBackward<R> {
    /// Create a new ExpandBackward
    pub fn new(
        input_id: TensorId,
        input_shape: Vec<usize>,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            input_shape,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for ExpandBackward<R>
where
    R::Client: RuntimeClient<R> + crate::ops::TensorOps<R> + ReduceOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());

        // Find dimensions that were broadcast and need to be summed
        let output_shape = grad_output.shape();
        let input_ndim = self.input_shape.len();
        let output_ndim = output_shape.len();

        let mut result = grad_output.clone();

        // Sum over extra leading dimensions
        if output_ndim > input_ndim {
            let extra_dims: Vec<usize> = (0..(output_ndim - input_ndim)).collect();
            result = client.sum(&result, &extra_dims, false)?;
        }

        // Sum over dimensions where input had size 1
        let offset = output_ndim.saturating_sub(input_ndim);
        let mut reduce_dims = Vec::new();
        for (i, &input_dim) in self.input_shape.iter().enumerate() {
            let output_idx = offset + i;
            if input_dim == 1 && output_shape[output_idx] > 1 {
                reduce_dims.push(i);
            }
        }

        if !reduce_dims.is_empty() {
            result = client.sum(&result, &reduce_dims, true)?;
        }

        // Reshape to input shape if needed
        if result.shape() != self.input_shape.as_slice() {
            result = result.reshape(&self.input_shape)?;
        }

        Ok(vec![Some(result)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        use crate::autograd::var_sum;

        let client = R::default_client(grad_output.tensor().device());

        let output_shape = grad_output.shape();
        let input_ndim = self.input_shape.len();
        let output_ndim = output_shape.len();

        let mut result = grad_output.clone();

        // Sum over extra leading dimensions using var_sum
        if output_ndim > input_ndim {
            let extra_dims: Vec<usize> = (0..(output_ndim - input_ndim)).collect();
            result = var_sum(&result, &extra_dims, false, &client)?;
        }

        // Sum over dimensions where input had size 1
        let offset = output_ndim.saturating_sub(input_ndim);
        let mut reduce_dims = Vec::new();
        for (i, &input_dim) in self.input_shape.iter().enumerate() {
            let output_idx = offset + i;
            if output_idx < output_shape.len() && input_dim == 1 && output_shape[output_idx] > 1 {
                reduce_dims.push(i);
            }
        }

        if !reduce_dims.is_empty() {
            result = var_sum(&result, &reduce_dims, true, &client)?;
        }

        // Reshape to input shape if needed using var_reshape
        if result.shape() != self.input_shape.as_slice() {
            result = var_reshape(&result, &self.input_shape)?;
        }

        Ok(vec![Some(result)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "ExpandBackward"
    }
}

// ============================================================================
// Var Operations for Shape
// ============================================================================

/// Reshape a Var to a new shape
///
/// Creates ReshapeBackward for gradient computation.
pub fn var_reshape<R: Runtime>(a: &Var<R>, shape: &[usize]) -> Result<Var<R>> {
    let output = a.tensor().reshape(shape)?;

    if a.requires_grad() {
        let grad_fn = ReshapeBackward::<R>::new(a.id(), a.shape().to_vec(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Transpose a Var (swap last two dimensions)
///
/// Creates TransposeBackward for gradient computation.
pub fn var_transpose<R: Runtime>(a: &Var<R>) -> Result<Var<R>> {
    let output = a.tensor().t()?;

    if a.requires_grad() {
        let grad_fn = TransposeBackward::<R>::new(a.id(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Permute a Var's dimensions
///
/// Creates PermuteBackward for gradient computation.
pub fn var_permute<R: Runtime>(a: &Var<R>, dims: &[usize]) -> Result<Var<R>> {
    let output = a.tensor().permute(dims)?;

    if a.requires_grad() {
        let grad_fn = PermuteBackward::<R>::new(a.id(), dims, a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Broadcast a Var to a new shape
///
/// Creates ExpandBackward for gradient computation.
pub fn var_broadcast_to<R: Runtime>(a: &Var<R>, shape: &[usize]) -> Result<Var<R>>
where
    R::Client: RuntimeClient<R> + crate::ops::TensorOps<R> + ReduceOps<R>,
{
    let output = a.tensor().broadcast_to(shape)?;

    if a.requires_grad() {
        let grad_fn = ExpandBackward::<R>::new(a.id(), a.shape().to_vec(), a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_reshape_backward() {
        let device = CpuDevice::new();

        // Input shape [2, 3], reshape to [3, 2]
        let input =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[3, 2], DType::F32, &device);

        let backward = ReshapeBackward::<CpuRuntime>::new(input.id(), vec![2, 3], None);
        let grads = backward.backward(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[2, 3]);
    }

    #[test]
    fn test_transpose_backward() {
        let device = CpuDevice::new();

        // Input shape [2, 3], transpose to [3, 2]
        let input =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[3, 2], DType::F32, &device);

        let backward = TransposeBackward::<CpuRuntime>::new(input.id(), None);
        let grads = backward.backward(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[2, 3]);
    }

    #[test]
    fn test_var_reshape() {
        let device = CpuDevice::new();

        let tensor =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let x = Var::new(tensor, true);

        let y = var_reshape(&x, &[3, 2]).unwrap();

        assert_eq!(y.shape(), &[3, 2]);
        assert!(y.requires_grad());
        assert!(y.grad_fn().is_some());
        assert_eq!(y.grad_fn().unwrap().name(), "ReshapeBackward");
    }

    #[test]
    fn test_var_transpose() {
        let device = CpuDevice::new();

        let tensor =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);
        let x = Var::new(tensor, true);

        let y = var_transpose(&x).unwrap();

        assert_eq!(y.shape(), &[3, 2]);
        assert!(y.requires_grad());
        assert!(y.grad_fn().is_some());
        assert_eq!(y.grad_fn().unwrap().name(), "TransposeBackward");
    }

    #[test]
    fn test_permute_backward() {
        let device = CpuDevice::new();

        // Input shape [2, 3, 4], permute to [3, 4, 2] (dims = [1, 2, 0])
        let input = Tensor::<CpuRuntime>::ones(&[2, 3, 4], DType::F32, &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[3, 4, 2], DType::F32, &device);

        let backward = PermuteBackward::<CpuRuntime>::new(input.id(), &[1, 2, 0], None);
        let grads = backward.backward(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        // Inverse permutation of [1, 2, 0] is [2, 0, 1]
        // So output shape [3, 4, 2] -> [2, 3, 4]
        assert_eq!(grad.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_var_permute() {
        let device = CpuDevice::new();

        let tensor = Tensor::<CpuRuntime>::ones(&[2, 3, 4], DType::F32, &device);
        let x = Var::new(tensor, true);

        let y = var_permute(&x, &[2, 0, 1]).unwrap();

        assert_eq!(y.shape(), &[4, 2, 3]);
        assert!(y.requires_grad());
        assert!(y.grad_fn().is_some());
        assert_eq!(y.grad_fn().unwrap().name(), "PermuteBackward");
    }

    #[test]
    fn test_expand_backward() {
        let device = CpuDevice::new();

        // Input shape [1, 3], expanded to [2, 3]
        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 3], DType::F32, &device);

        let backward = ExpandBackward::<CpuRuntime>::new(input.id(), vec![1, 3], None);
        let grads = backward.backward(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[1, 3]);

        // Each element should be summed over the broadcast dimension
        // grad_out is all 1s, broadcast dim had size 2, so sum = 2
        let grad_data: Vec<f32> = grad.to_vec();
        assert_eq!(grad_data, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_var_broadcast_to() {
        let device = CpuDevice::new();

        let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        let x = Var::new(tensor, true);

        let y = var_broadcast_to(&x, &[2, 3]).unwrap();

        assert_eq!(y.shape(), &[2, 3]);
        assert!(y.requires_grad());
        assert!(y.grad_fn().is_some());
        assert_eq!(y.grad_fn().unwrap().name(), "ExpandBackward");

        // Verify values are broadcast correctly (need contiguous for to_vec)
        let y_contiguous = y.tensor().contiguous();
        let y_data: Vec<f32> = y_contiguous.to_vec();
        assert_eq!(y_data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_reshape_backward_scalar() {
        let device = CpuDevice::new();

        // Scalar reshape to [1] and back
        let input = Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device);

        let backward = ReshapeBackward::<CpuRuntime>::new(input.id(), vec![], None);
        let grads = backward.backward(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[] as &[usize]);
    }

    #[test]
    fn test_transpose_backward_3d() {
        let device = CpuDevice::new();

        // 3D tensor transpose (swaps last two dims)
        let input = Tensor::<CpuRuntime>::ones(&[2, 3, 4], DType::F32, &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 4, 3], DType::F32, &device);

        let backward = TransposeBackward::<CpuRuntime>::new(input.id(), None);
        let grads = backward.backward(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_expand_backward_multiple_dims() {
        let device = CpuDevice::new();

        // Input shape [1, 1], expanded to [3, 4]
        let input = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1, 1], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[3, 4], DType::F32, &device);

        let backward = ExpandBackward::<CpuRuntime>::new(input.id(), vec![1, 1], None);
        let grads = backward.backward(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[1, 1]);

        // Sum over both dimensions: 3 * 4 = 12
        let grad_data: Vec<f32> = grad.to_vec();
        assert_eq!(grad_data, vec![12.0]);
    }

    #[test]
    fn test_permute_backward_identity() {
        let device = CpuDevice::new();

        // Identity permutation
        let input = Tensor::<CpuRuntime>::ones(&[2, 3], DType::F32, &device);
        let grad_out =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device);

        let backward = PermuteBackward::<CpuRuntime>::new(input.id(), &[0, 1], None);
        let grads = backward.backward(&grad_out).unwrap();

        let grad = grads[0].as_ref().unwrap();
        assert_eq!(grad.shape(), &[2, 3]);

        // Identity permutation should pass through unchanged
        let grad_data: Vec<f32> = grad.to_vec();
        assert_eq!(grad_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
