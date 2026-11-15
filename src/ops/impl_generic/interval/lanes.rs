//! Flat lane operands for native interval and expansion kernels.
//!
//! Each operand advances by zero or one element per output lane.
//! Other layouts become contiguous copies of their broadcast view first.

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;
use std::borrow::Cow;

/// One operand prepared for a flat lane loop.
#[derive(Debug)]
pub struct LaneOperand<'a, R: Runtime> {
    /// Tensor whose first element starts the walk. It keeps the storage alive.
    pub tensor: Cow<'a, Tensor<R>>,
    /// Element step per output lane: zero for one repeated element, one otherwise.
    pub step: u32,
}

impl<R: Runtime<DType = DType>> LaneOperand<'_, R> {
    /// Address of the first element read.
    pub fn addr(&self) -> u64 {
        self.tensor.ptr()
    }
}

/// Prepare `value` for a walk over `shape` in row-major order.
///
/// `materialize` copies a broadcast view into contiguous storage on the backend's stream.
pub fn lane_operand<'a, R, F>(
    value: &'a Tensor<R>,
    shape: &[usize],
    materialize: F,
) -> Result<LaneOperand<'a, R>>
where
    R: Runtime<DType = DType>,
    F: FnOnce(&Tensor<R>) -> Result<Tensor<R>>,
{
    if value.numel() == 1 {
        return Ok(LaneOperand {
            tensor: Cow::Borrowed(value),
            step: 0,
        });
    }
    if value.shape() == shape && value.is_contiguous() {
        return Ok(LaneOperand {
            tensor: Cow::Borrowed(value),
            step: 1,
        });
    }
    let view = value.broadcast_to(shape)?;
    Ok(LaneOperand {
        tensor: Cow::Owned(materialize(&view)?),
        step: 1,
    })
}

/// Element count of `shape`.
pub fn lane_count(shape: &[usize]) -> usize {
    shape.iter().product()
}
