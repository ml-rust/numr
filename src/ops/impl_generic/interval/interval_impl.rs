//! Composed interval operations shared by backends without native interval kernels.

use super::composed::{Composed, ComposedClient};
use super::interval_check::{IntervalBinary, check_binary};
use super::validate;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{IntervalEnclosure, IntervalOperand, IntervalPair};
use crate::runtime::{Device, Runtime};
use crate::tensor::Tensor;

/// Composed `any(mask != 0)` status update.
pub fn interval_flag_nonzero_impl<R, C>(
    client: &C,
    mask: &Tensor<R>,
    status: &Tensor<R>,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    validate::status(client, status)?;
    if !client.device().is_same(mask.device()) {
        return Err(Error::DeviceMismatch);
    }
    Composed::new(client, DType::U32, status)?.mark(mask)
}

/// Composed nonfinite status update.
pub fn interval_flag_nonfinite_impl<R, C>(
    client: &C,
    value: &Tensor<R>,
    status: &Tensor<R>,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    let dtype = validate::float_dtype(client, "interval_flag_nonfinite", &[value])?;
    validate::status(client, status)?;
    Composed::new(client, dtype, status)?.finite(value)
}

/// Composed bounds status update.
pub fn interval_flag_bounds_impl<R, C>(
    client: &C,
    value: IntervalOperand<'_, R>,
    status: &Tensor<R>,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    let dtype = validate::float_dtype(client, "interval_flag_bounds", &[value.lower, value.upper])?;
    validate::interval(value)?;
    validate::status(client, status)?;
    Composed::new(client, dtype, status)?
        .bounds(value.lower.clone(), value.upper.clone())
        .map(|_| ())
}

/// Composed endpoint snapshot with bounds status.
pub fn interval_capture_impl<R, C>(
    client: &C,
    value: IntervalOperand<'_, R>,
    status: &Tensor<R>,
) -> Result<IntervalPair<R>>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    let dtype = validate::float_dtype(client, "interval_capture", &[value.lower, value.upper])?;
    validate::interval(value)?;
    validate::status(client, status)?;
    let composed = Composed::new(client, dtype, status)?;
    let lower = composed.snapshot(value.lower)?;
    let upper = if validate::same_view(value.lower, value.upper) {
        lower.clone()
    } else {
        composed.snapshot(value.upper)?
    };
    composed.bounds(lower, upper)
}

/// Composed binary interval operation.
pub fn interval_binary_impl<R, C>(
    client: &C,
    op: IntervalBinary,
    a: IntervalOperand<'_, R>,
    b: IntervalOperand<'_, R>,
    enclosure: IntervalEnclosure,
    status: &Tensor<R>,
) -> Result<IntervalPair<R>>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    let dtype = check_binary(client, op.name(), a, b, status)?;
    let composed = Composed::new(client, dtype, status)?;
    let outward = match op {
        IntervalBinary::Add => composed.add_outward(a, b)?,
        IntervalBinary::Sub => composed.sub_outward(a, b)?,
        IntervalBinary::Mul => composed.mul_outward(a, b)?,
        IntervalBinary::DivPositive => composed.div_positive_outward(a, b.lower)?,
        IntervalBinary::DivPositiveInterval => composed.div_positive_interval_outward(a, b)?,
    };
    if enclosure == IntervalEnclosure::Outward {
        return Ok(outward);
    }
    let view = IntervalOperand {
        lower: &outward.0,
        upper: &outward.1,
    };
    match op {
        IntervalBinary::Add => composed.tighten_sum(a, b, view, false),
        IntervalBinary::Sub => composed.tighten_sum(a, b, view, true),
        IntervalBinary::Mul => {
            let tightened = if composed.fused() {
                composed.tighten_mul(a, b, view)?
            } else {
                outward.clone()
            };
            let view = IntervalOperand {
                lower: &tightened.0,
                upper: &tightened.1,
            };
            composed.restrict_mul_sign(a, b, view)
        }
        IntervalBinary::DivPositive | IntervalBinary::DivPositiveInterval => {
            let tightened = if composed.fused() {
                composed.tighten_div(a, b, view)?
            } else {
                outward.clone()
            };
            let view = IntervalOperand {
                lower: &tightened.0,
                upper: &tightened.1,
            };
            composed.restrict_div_sign(a, view)
        }
    }
}

/// Composed square roots of finite nonnegative intervals.
pub fn interval_sqrt_nonnegative_impl<R, C>(
    client: &C,
    a: IntervalOperand<'_, R>,
    status: &Tensor<R>,
) -> Result<IntervalPair<R>>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    let dtype = validate::float_dtype(client, "interval_sqrt_nonnegative", &[a.lower, a.upper])?;
    validate::interval(a)?;
    validate::status(client, status)?;
    Composed::new(client, dtype, status)?.sqrt_nonnegative(a)
}
