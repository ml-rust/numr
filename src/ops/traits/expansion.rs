//! Combined certified expansion kernels.

use super::interval_operand::{ExpansionOperand, ExpansionTriple, IntervalOperand, IntervalPair};
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Elementwise certified expansion arithmetic that runs each operation as one pass.
///
/// Status flags follow [`super::IntervalOps`]. Components require identical shapes.
/// Results equal the documented composition of ordinary rounded tensor operations bit for bit.
pub trait ExpansionOps<R: Runtime> {
    /// Snapshot `value` as an exact expansion with zero low component and zero radius.
    ///
    /// Sets `status` for nonfinite values.
    fn expansion_point(&self, value: &Tensor<R>, status: &Tensor<R>) -> Result<ExpansionTriple<R>>;

    /// Snapshot components and normalize the center with TwoSum.
    ///
    /// A zero low component keeps both center bit patterns.
    /// Sets `status` for nonfinite components or a negative radius.
    fn expansion_normalize(
        &self,
        value: ExpansionOperand<'_, R>,
        status: &Tensor<R>,
    ) -> Result<ExpansionTriple<R>>;

    /// Set `status` for nonfinite components or a negative radius.
    fn expansion_flag(&self, value: ExpansionOperand<'_, R>, status: &Tensor<R>) -> Result<()>;

    /// Add expansion centers with TwoSum compression and an outward radius sum.
    fn expansion_add(
        &self,
        a: ExpansionOperand<'_, R>,
        b: ExpansionOperand<'_, R>,
        status: &Tensor<R>,
    ) -> Result<ExpansionTriple<R>>;

    /// Subtract expansion centers with TwoSum compression and an outward radius sum.
    fn expansion_sub(
        &self,
        a: ExpansionOperand<'_, R>,
        b: ExpansionOperand<'_, R>,
        status: &Tensor<R>,
    ) -> Result<ExpansionTriple<R>>;

    /// Multiply expansions, enclosing four center products with fused residuals and both radii.
    ///
    /// An exact-one operand returns the other operand's component bits.
    /// The runtime must declare [`crate::runtime::FusedMulAddContract::Ieee754Fused`].
    fn expansion_mul(
        &self,
        a: ExpansionOperand<'_, R>,
        b: ExpansionOperand<'_, R>,
        status: &Tensor<R>,
    ) -> Result<ExpansionTriple<R>>;

    /// Project an expansion into an outward interval around its rounded center.
    fn expansion_interval(
        &self,
        value: ExpansionOperand<'_, R>,
        status: &Tensor<R>,
    ) -> Result<IntervalPair<R>>;

    /// Divide by an expansion whose projected lower bound is finite and strictly positive.
    ///
    /// The center is a corrected quotient. Its residual, enclosed with both input radii, sets the radius.
    /// Sets `status` for invalid divisors, which divide by one instead.
    /// The runtime must declare [`crate::runtime::FusedMulAddContract::Ieee754Fused`].
    fn expansion_div_positive(
        &self,
        a: ExpansionOperand<'_, R>,
        b: ExpansionOperand<'_, R>,
        status: &Tensor<R>,
    ) -> Result<ExpansionTriple<R>>;

    /// Lift an interval into its midpoint center and a radius that covers the whole interval.
    ///
    /// Equal endpoints keep the lower endpoint's bits and a zero radius.
    /// The runtime must declare [`crate::runtime::FloatArithmeticContract::Ieee754Basic`].
    fn expansion_from_interval(
        &self,
        value: IntervalOperand<'_, R>,
        status: &Tensor<R>,
    ) -> Result<ExpansionTriple<R>>;
}
