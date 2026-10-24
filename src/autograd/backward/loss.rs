use crate::autograd::Var;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Validate that the loss tensor is suitable for backward pass
///
/// Checks:
/// 1. Loss is a scalar (numel == 1)
/// 2. Loss requires gradients
#[inline]
pub(super) fn validate_loss<R: Runtime>(loss: &Var<R>, fn_name: &str) -> Result<()> {
    if loss.numel() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![1],
            got: loss.shape().to_vec(),
        });
    }

    if !loss.requires_grad() {
        return Err(Error::Internal(format!(
            "{}() called on tensor that doesn't require grad",
            fn_name
        )));
    }

    Ok(())
}

/// Create the initial gradient tensor for the loss (dL/dL = 1)
#[inline]
pub(super) fn create_loss_gradient<R: Runtime<DType = DType>>(loss: &Var<R>) -> Result<Tensor<R>> {
    Tensor::<R>::ones(loss.shape(), loss.tensor().dtype(), loss.tensor().device())
}
