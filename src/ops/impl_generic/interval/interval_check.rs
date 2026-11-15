//! Interval operation selectors and argument checks shared by every backend.

use super::validate;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::IntervalOperand;
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Binary interval operation selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntervalBinary {
    /// `a + b`.
    Add,
    /// `a - b`.
    Sub,
    /// `a * b`.
    Mul,
    /// `a / d` with a represented divisor passed as both endpoints of `b`.
    DivPositive,
    /// `a / b` with a strictly positive interval divisor.
    DivPositiveInterval,
}

impl IntervalBinary {
    /// Operation name for errors.
    pub fn name(self) -> &'static str {
        match self {
            Self::Add => "interval_add",
            Self::Sub => "interval_sub",
            Self::Mul => "interval_mul",
            Self::DivPositive => "interval_div_positive",
            Self::DivPositiveInterval => "interval_div_positive_interval",
        }
    }
}

/// Output shape, with the broadcast error the composed operation reports first.
pub fn binary_shape<R: Runtime<DType = DType>>(
    op: IntervalBinary,
    a: IntervalOperand<'_, R>,
    b: IntervalOperand<'_, R>,
) -> Result<Vec<usize>> {
    match op {
        IntervalBinary::Add | IntervalBinary::Sub | IntervalBinary::Mul => {
            validate::broadcast(a.lower, b.lower)
        }
        IntervalBinary::DivPositive | IntervalBinary::DivPositiveInterval => {
            validate::broadcast(b.lower, a.lower)
        }
    }
}

/// Cleared U32 scalar status flag.
pub fn interval_status_impl<R, C>(client: &C) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    Tensor::zeros(&[], DType::U32, client.device())
}

/// Validate a binary interval call and return its dtype.
pub fn check_binary<R, C>(
    client: &C,
    op: &'static str,
    a: IntervalOperand<'_, R>,
    b: IntervalOperand<'_, R>,
    status: &Tensor<R>,
) -> Result<DType>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    let dtype = validate::float_dtype(client, op, &[a.lower, a.upper, b.lower, b.upper])?;
    validate::interval(a)?;
    validate::interval(b)?;
    validate::status(client, status)?;
    Ok(dtype)
}
