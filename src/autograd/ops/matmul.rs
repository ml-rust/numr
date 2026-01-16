//! Backward implementation for matrix multiplication
//!
//! Implements gradient computation for matmul: C = A @ B

use crate::autograd::GradFn;
use crate::error::Result;
use crate::ops::TensorOps;
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

// ============================================================================
// MatmulBackward
// ============================================================================

/// Backward for matrix multiplication: C = A @ B
///
/// For C = A @ B where A is (m, k) and B is (k, n):
/// - dL/dA = dL/dC @ B^T  (shape: m x k)
/// - dL/dB = A^T @ dL/dC  (shape: k x n)
///
/// Supports batched matmul as well.
pub struct MatmulBackward<R: Runtime> {
    input_ids: [TensorId; 2],
    saved_tensors: Vec<Tensor<R>>, // [a, b]
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 2],
}

impl<R: Runtime> MatmulBackward<R> {
    /// Create a new MatmulBackward
    pub fn new(
        a_id: TensorId,
        b_id: TensorId,
        a: Tensor<R>,
        b: Tensor<R>,
        a_grad_fn: Option<Arc<dyn GradFn<R>>>,
        b_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [a_id, b_id],
            saved_tensors: vec![a, b],
            input_grad_fns: [a_grad_fn, b_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for MatmulBackward<R>
where
    R::Client: TensorOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let saved_a = &self.saved_tensors[0];
        let saved_b = &self.saved_tensors[1];

        // C = A @ B
        // dL/dA = dL/dC @ B^T
        // dL/dB = A^T @ dL/dC

        // Transpose B: swap last two dimensions
        let b_t = saved_b.t()?;
        let grad_a = client.matmul(grad_output, &b_t)?;

        // Transpose A: swap last two dimensions
        let a_t = saved_a.t()?;
        let grad_b = client.matmul(&a_t, grad_output)?;

        Ok(vec![Some(grad_a), Some(grad_b)])
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
        "MatmulBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_matmul_backward_2x2() {
        let device = CpuDevice::new();

        // A = [[1, 2], [3, 4]] (2x2)
        // B = [[5, 6], [7, 8]] (2x2)
        // C = A @ B = [[19, 22], [43, 50]]
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device);

        // Assume dL/dC = [[1, 1], [1, 1]] (ones)
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 2], DType::F32, &device);

        let backward =
            MatmulBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), b.clone(), None, None);
        let grads = backward.backward(&grad_out).unwrap();

        // dL/dA = dL/dC @ B^T
        // B^T = [[5, 7], [6, 8]]
        // dL/dA = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]] = [[11, 15], [11, 15]]
        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert_eq!(grad_a, vec![11.0, 15.0, 11.0, 15.0]);

        // dL/dB = A^T @ dL/dC
        // A^T = [[1, 3], [2, 4]]
        // dL/dB = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]] = [[4, 4], [6, 6]]
        let grad_b: Vec<f32> = grads[1].as_ref().unwrap().to_vec();
        assert_eq!(grad_b, vec![4.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn test_matmul_backward_non_square() {
        let device = CpuDevice::new();

        // A = [[1, 2, 3]] (1x3)
        // B = [[4], [5], [6]] (3x1)
        // C = A @ B = [[32]] (1x1)
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0, 6.0], &[3, 1], &device);

        // dL/dC = [[1]]
        let grad_out = Tensor::<CpuRuntime>::ones(&[1, 1], DType::F32, &device);

        let backward =
            MatmulBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), b.clone(), None, None);
        let grads = backward.backward(&grad_out).unwrap();

        // dL/dA = dL/dC @ B^T = [[1]] @ [[4, 5, 6]] = [[4, 5, 6]]
        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[1, 3]);
        let grad_a_data: Vec<f32> = grad_a.to_vec();
        assert_eq!(grad_a_data, vec![4.0, 5.0, 6.0]);

        // dL/dB = A^T @ dL/dC = [[1], [2], [3]] @ [[1]] = [[1], [2], [3]]
        let grad_b = grads[1].as_ref().unwrap();
        assert_eq!(grad_b.shape(), &[3, 1]);
        let grad_b_data: Vec<f32> = grad_b.to_vec();
        assert_eq!(grad_b_data, vec![1.0, 2.0, 3.0]);
    }
}
