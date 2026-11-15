//! Reference expansion quotient and interval lift.

use super::composed::{Composed, ComposedClient};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ExpansionOperand, ExpansionTriple, IntervalOperand};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

fn parts<R: Runtime<DType = DType>>(value: &ExpansionTriple<R>) -> ExpansionOperand<'_, R> {
    ExpansionOperand {
        high: &value.0,
        low: &value.1,
        abs_error: &value.2,
    }
}

fn point<R: Runtime<DType = DType>>(value: &Tensor<R>) -> IntervalOperand<'_, R> {
    IntervalOperand {
        lower: value,
        upper: value,
    }
}

impl<R, C> Composed<'_, R, C>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    fn owned_flagged(&self, value: ExpansionTriple<R>) -> Result<ExpansionTriple<R>> {
        self.expansion_flag(parts(&value))?;
        Ok(value)
    }

    /// Corrected quotient center certified through an independent residual enclosure.
    pub fn expansion_div_positive(
        &self,
        a: ExpansionOperand<'_, R>,
        b: ExpansionOperand<'_, R>,
    ) -> Result<ExpansionTriple<R>> {
        let c = self.client;
        let (den_lower, den_upper) = self.expansion_interval(b)?;
        let zero = Tensor::zeros(a.high.shape(), self.dtype, c.device())?;
        let one = Tensor::ones(a.high.shape(), self.dtype, c.device())?;
        let positive = c.gt(&den_lower, &zero)?;
        self.mark(&c.eq(&positive, &zero)?)?;
        let valid = c.mul(
            &c.mul(&positive, &self.finite_mask(&den_lower)?)?,
            &c.mul(&self.finite_mask(&den_upper)?, &self.finite_mask(b.high)?)?,
        )?;
        let valid = c.mul(&valid, &c.gt(b.high, &zero)?)?;
        self.mark(&c.eq(&valid, &zero)?)?;
        // Mask invalid denominator lanes before eager quotient evaluation.
        let denominator = c.where_cond(&valid, b.high, &one)?;
        let numerator = c.where_cond(&valid, a.high, &zero)?;
        let q0 = c.div(&numerator, &denominator)?;
        let provisional = self.owned_flagged((q0, zero.clone(), zero.clone()))?;
        let provisional_product = self.expansion_mul(parts(&provisional), b)?;
        let first_residual = self.expansion_sum(a, parts(&provisional_product), true)?;
        let residual_high = c.where_cond(&valid, &first_residual.0, &zero)?;
        let q1 = c.div(&residual_high, &denominator)?;
        self.finite(&q1)?;
        let (high, low) = c.two_sum(&provisional.0, &q1)?;
        let candidate = self.owned_flagged((high, low, zero.clone()))?;
        let candidate_product = self.expansion_mul(parts(&candidate), b)?;
        let residual = self.expansion_sum(a, parts(&candidate_product), true)?;
        let (residual_lower, residual_upper) = self.expansion_interval(parts(&residual))?;
        let magnitude = c.maximum(&c.abs(&residual_lower)?, &c.abs(&residual_upper)?)?;
        self.point(&magnitude)?;
        let lower = c.where_cond(&valid, &den_lower, &one)?;
        let (_, radius) = self.div_positive_outward(point(&magnitude), &lower)?;
        self.owned_flagged((candidate.0, candidate.1, radius))
    }

    /// Lift a complete interval into an expansion center and radius.
    pub fn expansion_from_interval(
        &self,
        value: IntervalOperand<'_, R>,
    ) -> Result<ExpansionTriple<R>> {
        let c = self.client;
        let (lower, upper) = self.bounds(value.lower.clone(), value.upper.clone())?;
        let equal = c.eq(&lower, &upper)?;
        let midpoint = c.midpoint(&lower, &upper)?;
        let center = c.where_cond(&equal, &lower, &midpoint)?;
        let zero = Tensor::zeros(center.shape(), center.dtype(), c.device())?;
        self.point(&center)?;
        self.point(&lower)?;
        self.point(&upper)?;
        let lower_distance = self.sub_tightened(point(&center), point(&lower))?;
        let upper_distance = self.sub_tightened(point(&upper), point(&center))?;
        let radius = c.maximum(&lower_distance.1, &upper_distance.1)?;
        let radius = c.where_cond(&equal, &zero, &radius)?;
        self.expansion_normalize(ExpansionOperand {
            high: &center,
            low: &zero,
            abs_error: &radius,
        })
    }

    fn sub_tightened(
        &self,
        a: IntervalOperand<'_, R>,
        b: IntervalOperand<'_, R>,
    ) -> Result<crate::ops::IntervalPair<R>> {
        let (lower, upper) = self.sub_outward(a, b)?;
        let outward = IntervalOperand {
            lower: &lower,
            upper: &upper,
        };
        self.tighten_sum(a, b, outward, true)
    }
}
