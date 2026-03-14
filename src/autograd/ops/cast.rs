//! Backward implementation for dtype cast operation
//!
//! The backward of cast(x, target_dtype) is cast(grad_output, input_dtype).

use crate::autograd::GradFn;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::TypeConversionOps;
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};

/// Backward for cast: z = cast(a, target_dtype)
///
/// Gradient: dL/da = cast(dL/dz, a.dtype)
pub struct CastBackward<R: Runtime> {
    input_id: TensorId,
    input_dtype: DType,
    _marker: std::marker::PhantomData<R>,
}

impl<R: Runtime> CastBackward<R> {
    /// Create a new CastBackward
    pub fn new(input_id: TensorId, input_dtype: DType) -> Self {
        Self {
            input_id,
            input_dtype,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for CastBackward<R>
where
    R::Client: TypeConversionOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let grad = if grad_output.dtype() == self.input_dtype {
            grad_output.clone()
        } else {
            client.cast(grad_output, self.input_dtype)?
        };
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn name(&self) -> &'static str {
        "CastBackward"
    }
}
