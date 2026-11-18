//! Composed expansion operations shared by backends without native expansion kernels.

use super::composed::{Composed, ComposedClient};
use super::expansion_check::{ExpansionBinary, check_expansions, require_basic, require_fused};
use super::validate;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ExpansionOperand, ExpansionTriple, IntervalOperand, IntervalPair};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Composed exact expansion of represented values.
pub fn expansion_point_impl<R, C>(
    client: &C,
    value: &Tensor<R>,
    status: &Tensor<R>,
) -> Result<ExpansionTriple<R>>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    let dtype = validate::float_dtype(client, "expansion_point", &[value])?;
    validate::status(client, status)?;
    Composed::new(client, dtype, status)?.expansion_point(value)
}

/// Composed TwoSum normalization of supplied components.
pub fn expansion_normalize_impl<R, C>(
    client: &C,
    value: ExpansionOperand<'_, R>,
    status: &Tensor<R>,
) -> Result<ExpansionTriple<R>>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    let dtype = check_expansions(client, "expansion_normalize", &[value], status)?;
    Composed::new(client, dtype, status)?.expansion_normalize(value)
}

/// Composed component status update.
pub fn expansion_flag_impl<R, C>(
    client: &C,
    value: ExpansionOperand<'_, R>,
    status: &Tensor<R>,
) -> Result<()>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    let dtype = check_expansions(client, "expansion_flag", &[value], status)?;
    Composed::new(client, dtype, status)?.expansion_flag(value)
}

/// Composed binary expansion operation.
pub fn expansion_binary_impl<R, C>(
    client: &C,
    op: ExpansionBinary,
    a: ExpansionOperand<'_, R>,
    b: ExpansionOperand<'_, R>,
    status: &Tensor<R>,
) -> Result<ExpansionTriple<R>>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    let dtype = check_expansions(client, op.name(), &[a, b], status)?;
    let composed = Composed::new(client, dtype, status)?;
    match op {
        ExpansionBinary::Add => composed.expansion_sum(a, b, false),
        ExpansionBinary::Sub => composed.expansion_sum(a, b, true),
        ExpansionBinary::Mul => {
            require_fused::<R>(dtype, op.name())?;
            composed.expansion_mul(a, b)
        }
        ExpansionBinary::DivPositive => {
            require_fused::<R>(dtype, op.name())?;
            composed.expansion_div_positive(a, b)
        }
    }
}

/// Composed outward projection of an expansion.
pub fn expansion_interval_impl<R, C>(
    client: &C,
    value: ExpansionOperand<'_, R>,
    status: &Tensor<R>,
) -> Result<IntervalPair<R>>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    let dtype = check_expansions(client, "expansion_interval", &[value], status)?;
    Composed::new(client, dtype, status)?.expansion_interval(value)
}

/// Composed lift of an interval into an expansion.
pub fn expansion_from_interval_impl<R, C>(
    client: &C,
    value: IntervalOperand<'_, R>,
    status: &Tensor<R>,
) -> Result<ExpansionTriple<R>>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    let dtype = validate::float_dtype(
        client,
        "expansion_from_interval",
        &[value.lower, value.upper],
    )?;
    validate::interval(value)?;
    validate::status(client, status)?;
    require_basic::<R>(dtype, "expansion_from_interval")?;
    Composed::new(client, dtype, status)?.expansion_from_interval(value)
}
