//! Expansion operation selectors and argument checks shared by every backend.

use super::validate;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ExpansionOperand;
use crate::runtime::{FloatArithmeticContract, FusedMulAddContract, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Expansion operation selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpansionBinary {
    /// Center sum.
    Add,
    /// Center difference.
    Sub,
    /// Center product.
    Mul,
    /// Corrected quotient by a positive expansion.
    DivPositive,
}

impl ExpansionBinary {
    /// Whether the operation needs IEEE fused multiply-add.
    pub fn needs_fused(self) -> bool {
        matches!(self, Self::Mul | Self::DivPositive)
    }

    /// Operation name for errors.
    pub fn name(self) -> &'static str {
        match self {
            Self::Add => "expansion_add",
            Self::Sub => "expansion_sub",
            Self::Mul => "expansion_mul",
            Self::DivPositive => "expansion_div_positive",
        }
    }
}

/// Validate expansion operands and return their dtype.
pub fn check_expansions<R, C>(
    client: &C,
    op: &'static str,
    values: &[ExpansionOperand<'_, R>],
    status: &Tensor<R>,
) -> Result<DType>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    let components: Vec<&Tensor<R>> = values
        .iter()
        .flat_map(|value| [value.high, value.low, value.abs_error])
        .collect();
    let dtype = validate::float_dtype(client, op, &components)?;
    validate::expansions(values)?;
    validate::status(client, status)?;
    Ok(dtype)
}

/// Reject runtimes without IEEE fused multiply-add for `dtype`.
pub fn require_fused<R: Runtime<DType = DType>>(dtype: DType, op: &'static str) -> Result<()> {
    if R::fused_mul_add_contract(dtype) != FusedMulAddContract::Ieee754Fused {
        return Err(Error::InvalidArgument {
            arg: "runtime",
            reason: format!(
                "{op} needs IEEE fused multiply-add for {dtype:?}. Use a runtime that declares it."
            ),
        });
    }
    Ok(())
}

/// Reject runtimes without IEEE basic arithmetic for `dtype`, naming `op`.
pub fn require_basic<R: Runtime<DType = DType>>(dtype: DType, op: &'static str) -> Result<()> {
    let contract = R::float_arithmetic_contract(dtype);
    if contract != FloatArithmeticContract::Ieee754Basic {
        return Err(Error::backend_limitation(
            R::name(),
            op,
            format!(
                "dtype={dtype:?}, contract={contract:?}. Select a runtime providing IEEE 754 basic arithmetic"
            ),
        ));
    }
    Ok(())
}
