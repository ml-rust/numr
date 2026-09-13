//! Backward implementation for `matmul_wide`: C = A @ B with C in the
//! accumulator dtype.
//!
//! The forward widens F16/BF16 to F32 (and I8 to I32), so the incoming
//! gradient arrives in the output dtype while the operands' gradients must be
//! in the operands' dtype. Casting the gradient to the operand dtype first,
//! then applying the `matmul` formulas, gives `grad @ B^T` and `A^T @ grad`
//! in the dtype the operands carry.

use super::matmul::MatmulBackward;
use crate::autograd::var_ops::var_cast;
use crate::autograd::{GradFn, Var};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{MatmulOps, TensorOps, TypeConversionOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Backward for `matmul_wide`: the `matmul` backward applied to the gradient
/// cast back to the operand dtype.
pub struct MatmulWideBackward<R: Runtime> {
    inner: MatmulBackward<R>,
    operand_dtype: DType,
}

impl<R: Runtime<DType = DType>> MatmulWideBackward<R> {
    /// Create a new MatmulWideBackward. `a` and `b` share a dtype, which is
    /// the dtype both gradients are produced in.
    pub fn new(
        a_id: TensorId,
        b_id: TensorId,
        a: Tensor<R>,
        b: Tensor<R>,
        a_grad_fn: Option<Arc<dyn GradFn<R>>>,
        b_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        let operand_dtype = a.dtype();
        Self {
            inner: MatmulBackward::new(a_id, b_id, a, b, a_grad_fn, b_grad_fn),
            operand_dtype,
        }
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for MatmulWideBackward<R>
where
    R::Client: MatmulOps<R> + TensorOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>, needed: &[bool]) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        // A non-widening dtype gets its gradient in its own dtype already;
        // the cast is a no-op there and skipped.
        let grad = if grad_output.dtype() == self.operand_dtype {
            grad_output.clone()
        } else {
            client.cast(grad_output, self.operand_dtype)?
        };
        self.inner.backward(&grad, needed)
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + MatmulOps<R> + TensorOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());
        // `var_cast` keeps the graph through the cast, so second-order
        // gradients flow back into the F32 gradient.
        let grad = var_cast(grad_output, self.operand_dtype, &client)?;
        self.inner.backward_var(&grad)
    }

    fn inputs(&self) -> &[TensorId] {
        self.inner.inputs()
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.inner.input_grad_fns()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        self.inner.saved_tensors()
    }

    fn name(&self) -> &'static str {
        "MatmulWideBackward"
    }
}

#[cfg(all(test, feature = "f16"))]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    /// F16 operands, an F32 gradient: both gradients come back in F16 and
    /// carry the `matmul` values.
    #[test]
    fn test_matmul_wide_backward_casts_grad_to_operand_dtype() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a32 =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
        let b32 =
            Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device).unwrap();
        let a = client.cast(&a32, DType::F16).unwrap();
        let b = client.cast(&b32, DType::F16).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 2], DType::F32, &device).unwrap();

        let backward =
            MatmulWideBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), b.clone(), None, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.dtype(), DType::F16);
        let grad_a: Vec<f32> = client.cast(grad_a, DType::F32).unwrap().to_vec();
        assert_eq!(grad_a, vec![11.0, 15.0, 11.0, 15.0]);

        let grad_b = grads[1].as_ref().unwrap();
        assert_eq!(grad_b.dtype(), DType::F16);
        let grad_b: Vec<f32> = client.cast(grad_b, DType::F32).unwrap().to_vec();
        assert_eq!(grad_b, vec![4.0, 4.0, 6.0, 6.0]);
    }

    /// The `needed` mask reaches the inner backward: an unwanted slot stays
    /// `None`.
    #[test]
    fn test_matmul_wide_backward_honours_needed_mask() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a32 =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
        let a = client.cast(&a32, DType::BF16).unwrap();
        let b = client.cast(&a32, DType::BF16).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 2], DType::F32, &device).unwrap();

        let backward =
            MatmulWideBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), b.clone(), None, None);
        let grads = backward.backward(&grad_out, &[false, true]).unwrap();
        assert!(grads[0].is_none());
        assert_eq!(grads[1].as_ref().unwrap().dtype(), DType::BF16);
    }
}
