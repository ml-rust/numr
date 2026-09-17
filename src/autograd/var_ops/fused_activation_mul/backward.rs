//! Backward pass for fused activation-mul: `output = activation(a) * b`
//!
//! Gradients:
//! - d_b = grad_output * activation(a)
//! - d_a = grad_output * b * activation'(a)
//!
//! Derivatives:
//! - silu'(x)    = sigmoid(x) * (1 + x - silu(x))
//! - gelu'(x)    = 0.5*(1+tanh(inner)) + 0.5*x*sech²(inner)*sqrt(2/π)*(1+3*0.044715*x²)
//! - relu'(x)    = 1 if x > 0, else 0
//! - sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

use super::derivative::compute_activation_derivative;
use super::fused_kind::FusedKind;
use crate::autograd::Var;
use crate::autograd::var_ops::var_mul;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ScalarOps, TensorOps, UnaryOps,
};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

pub struct FusedActivationMulBackward<R: Runtime> {
    input_ids: [crate::tensor::TensorId; 2],
    saved_a: crate::tensor::Tensor<R>,
    saved_b: crate::tensor::Tensor<R>,
    saved_activation_a: crate::tensor::Tensor<R>,
    kind: FusedKind,
    a_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    b_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
}

impl<R: Runtime> FusedActivationMulBackward<R> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        a_id: crate::tensor::TensorId,
        b_id: crate::tensor::TensorId,
        a: crate::tensor::Tensor<R>,
        b: crate::tensor::Tensor<R>,
        activation_a: crate::tensor::Tensor<R>,
        kind: FusedKind,
        a_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
        b_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [a_id, b_id],
            saved_a: a,
            saved_b: b,
            saved_activation_a: activation_a,
            kind,
            a_grad_fn,
            b_grad_fn,
        }
    }
}

impl<R: Runtime<DType = DType>> crate::autograd::GradFn<R> for FusedActivationMulBackward<R>
where
    R::Client: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
{
    fn backward(
        &self,
        grad_output: &crate::tensor::Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<crate::tensor::Tensor<R>>>> {
        // One fused kernel returns both gradients together, so neither slot has
        // a cost of its own to skip. Guarding would need the kernel to take the
        // mask.
        let client = R::default_client(grad_output.device());

        // Delegate to fused backward trait method — allows backends (e.g. CUDA)
        // to provide a single fused kernel for the entire backward pass.
        let (d_a, d_b) = match self.kind {
            FusedKind::Silu => client.silu_mul_bwd(grad_output, &self.saved_a, &self.saved_b)?,
            FusedKind::Gelu => client.gelu_mul_bwd(grad_output, &self.saved_a, &self.saved_b)?,
            FusedKind::Relu => client.relu_mul_bwd(grad_output, &self.saved_a, &self.saved_b)?,
            FusedKind::Sigmoid => {
                client.sigmoid_mul_bwd(grad_output, &self.saved_a, &self.saved_b)?
            }
        };

        Ok(vec![Some(d_a), Some(d_b)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R>
            + TensorOps<R>
            + ActivationOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + UnaryOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());

        // d_b = grad_output * activation(a) (activation_a is constant w.r.t. higher-order)
        let act_var = Var::new(self.saved_activation_a.clone(), false);
        let d_b = var_mul(grad_output, &act_var, &client)?;

        // d_a = grad_output * b * activation'(a)
        let activation_deriv = compute_activation_derivative(
            &client,
            &self.saved_a,
            &self.saved_activation_a,
            self.kind,
        )?;
        let deriv_var = Var::new(activation_deriv, false);
        let b_var = Var::new(self.saved_b.clone(), false);
        let grad_times_b = var_mul(grad_output, &b_var, &client)?;
        let d_a = var_mul(&grad_times_b, &deriv_var, &client)?;

        Ok(vec![Some(d_a), Some(d_b)])
    }

    fn inputs(&self) -> &[crate::tensor::TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn crate::autograd::GradFn<R>>>> {
        vec![self.a_grad_fn.clone(), self.b_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[crate::tensor::Tensor<R>] {
        std::slice::from_ref(&self.saved_a)
    }

    fn name(&self) -> &'static str {
        match self.kind {
            FusedKind::Silu => "SiluMulBackward",
            FusedKind::Gelu => "GeluMulBackward",
            FusedKind::Relu => "ReluMulBackward",
            FusedKind::Sigmoid => "SigmoidMulBackward",
        }
    }
}
