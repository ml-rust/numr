//! Backward pass for differentiable 1D convolution.

use crate::autograd::Var;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, ConvOps, PaddingMode, ReduceOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

use super::grad_math::{conv1d_input_backward, conv1d_weight_backward};

/// Backward for conv1d.
///
/// Computes gradients for input, weight, and bias using:
/// - d_input: transposed convolution (conv with flipped kernel, adjusted padding)
/// - d_weight: cross-correlation of input with grad_output
/// - d_bias: sum of grad_output over batch and spatial dims
pub struct Conv1dBackward<R: Runtime> {
    input_ids: Vec<crate::tensor::TensorId>,
    saved_input: crate::tensor::Tensor<R>,
    saved_weight: crate::tensor::Tensor<R>,
    input_shape: Vec<usize>,
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
    groups: usize,
    input_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    weight_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    bias_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
}

impl<R: Runtime> Conv1dBackward<R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_id: crate::tensor::TensorId,
        weight_id: crate::tensor::TensorId,
        bias_id: Option<crate::tensor::TensorId>,
        input: crate::tensor::Tensor<R>,

        weight: crate::tensor::Tensor<R>,
        input_shape: Vec<usize>,
        stride: usize,
        padding: PaddingMode,
        dilation: usize,
        groups: usize,
        input_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
        weight_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
        bias_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    ) -> Self {
        let mut ids = vec![input_id, weight_id];
        if let Some(bid) = bias_id {
            ids.push(bid);
        }
        Self {
            input_ids: ids,
            saved_input: input,
            saved_weight: weight,
            input_shape,
            stride,
            padding,
            dilation,
            groups,
            input_grad_fn,
            weight_grad_fn,
            bias_grad_fn,
        }
    }
}

impl<R: Runtime<DType = DType>> crate::autograd::GradFn<R> for Conv1dBackward<R>
where
    R::Client: ConvOps<R> + TensorOps<R> + ReduceOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    fn backward(
        &self,
        grad_output: &crate::tensor::Tensor<R>,
        needed: &[bool],
    ) -> Result<Vec<Option<crate::tensor::Tensor<R>>>> {
        let client = R::default_client(grad_output.device());

        // The input gradient and the weight gradient are separate convolutions
        // sharing nothing but `grad_output`, so each is guarded. A frozen conv
        // layer skips the whole cross-correlation that builds d_weight.

        // d_input via transposed convolution
        let d_input = if needed[0] {
            Some(conv1d_input_backward::<R, _>(
                &client,
                grad_output,
                &self.saved_weight,
                &self.input_shape,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )?)
        } else {
            None
        };

        // d_weight via cross-correlation
        let d_weight = if needed[1] {
            Some(conv1d_weight_backward::<R, _>(
                &client,
                grad_output,
                &self.saved_input,
                self.saved_weight.shape(),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )?)
        } else {
            None
        };

        // d_bias = sum over batch and length dims
        let d_bias = if self.input_ids.len() > 2 && needed[2] {
            // grad_output shape: [batch, c_out, output_len]
            // sum over dim 0 (batch) and dim 2 (length) → [c_out]
            let summed = client.sum(grad_output, &[0, 2], false)?;
            Some(summed)
        } else {
            None
        };

        Ok(vec![d_input, d_weight, d_bias])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R>
            + ConvOps<R>
            + TensorOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + ScalarOps<R>,
    {
        // First-order only for conv — second-order conv is rarely needed
        // Second-order traversal keeps every node, so ask for every gradient.
        let grads = self.backward_all(grad_output.tensor())?;
        Ok(grads
            .into_iter()
            .map(|g| g.map(|t| Var::new(t, true)))
            .collect())
    }

    fn inputs(&self) -> &[crate::tensor::TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn crate::autograd::GradFn<R>>>> {
        let mut fns = vec![self.input_grad_fn.clone(), self.weight_grad_fn.clone()];
        if self.input_ids.len() > 2 {
            fns.push(self.bias_grad_fn.clone());
        }
        fns
    }

    fn saved_tensors(&self) -> &[crate::tensor::Tensor<R>] {
        std::slice::from_ref(&self.saved_input)
    }

    fn name(&self) -> &'static str {
        "Conv1dBackward"
    }
}
