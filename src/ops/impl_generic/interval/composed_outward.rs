//! Reference outward interval operations with exact-zero and exact-one identities.

use super::composed::{Composed, ComposedClient};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{IntervalOperand, IntervalPair};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

impl<R, C> Composed<'_, R, C>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    /// Outward sum.
    pub fn add_outward(
        &self,
        a: IntervalOperand<'_, R>,
        b: IntervalOperand<'_, R>,
    ) -> Result<IntervalPair<R>> {
        let c = self.client;
        let a_zero = self.zero_mask(a)?;
        let b_zero = self.zero_mask(b)?;
        let lower = c.next_down(&c.add(a.lower, b.lower)?)?;
        let upper = c.next_up(&c.add(a.upper, b.upper)?)?;
        let lower = c.where_cond(&a_zero, b.lower, &c.where_cond(&b_zero, a.lower, &lower)?)?;
        let upper = c.where_cond(&a_zero, b.upper, &c.where_cond(&b_zero, a.upper, &upper)?)?;
        self.bounds(lower, upper)
    }

    /// Outward difference.
    pub fn sub_outward(
        &self,
        a: IntervalOperand<'_, R>,
        b: IntervalOperand<'_, R>,
    ) -> Result<IntervalPair<R>> {
        let c = self.client;
        let a_zero = self.zero_mask(a)?;
        let b_zero = self.zero_mask(b)?;
        let lower = c.next_down(&c.sub(a.lower, b.upper)?)?;
        let upper = c.next_up(&c.sub(a.upper, b.lower)?)?;
        let lower = c.where_cond(
            &b_zero,
            a.lower,
            &c.where_cond(&a_zero, &c.neg(b.upper)?, &lower)?,
        )?;
        let upper = c.where_cond(
            &b_zero,
            a.upper,
            &c.where_cond(&a_zero, &c.neg(b.lower)?, &upper)?,
        )?;
        self.bounds(lower, upper)
    }

    /// Outward product over all four endpoint products.
    pub fn mul_outward(
        &self,
        a: IntervalOperand<'_, R>,
        b: IntervalOperand<'_, R>,
    ) -> Result<IntervalPair<R>> {
        let c = self.client;
        let a_zero = self.zero_mask(a)?;
        let b_zero = self.zero_mask(b)?;
        let zero_product = c.maximum(&a_zero, &b_zero)?;
        // Mask before multiplication, so discarded zero branches never evaluate zero times infinity.
        let a_lower = c.where_cond(&zero_product, &self.zero, a.lower)?;
        let a_upper = c.where_cond(&zero_product, &self.zero, a.upper)?;
        let b_lower = c.where_cond(&zero_product, &self.one, b.lower)?;
        let b_upper = c.where_cond(&zero_product, &self.one, b.upper)?;
        let (lower, upper) = self.corner_extrema(
            c.mul(&a_lower, &b_lower)?,
            c.mul(&a_lower, &b_upper)?,
            c.mul(&a_upper, &b_lower)?,
            c.mul(&a_upper, &b_upper)?,
        )?;
        let exact_lower = c.where_cond(&a_zero, a.lower, b.lower)?;
        let exact_upper = c.where_cond(&a_zero, a.upper, b.upper)?;
        let lower = c.where_cond(&zero_product, &exact_lower, &lower)?;
        let upper = c.where_cond(&zero_product, &exact_upper, &upper)?;
        self.bounds(lower, upper)
    }

    /// Outward quotient by represented divisors.
    pub fn div_positive_outward(
        &self,
        a: IntervalOperand<'_, R>,
        divisor: &Tensor<R>,
    ) -> Result<IntervalPair<R>> {
        let c = self.client;
        let valid = self.div_positive_inputs(divisor)?;
        let denominator = c.where_cond(&valid, divisor, &self.one)?;
        let identity = c.eq(divisor, &self.one)?;
        let a_lower = c.where_cond(&identity, &self.one, a.lower)?;
        let a_upper = c.where_cond(&identity, &self.one, a.upper)?;
        let denominator = c.where_cond(&identity, &self.one, &denominator)?;
        let lower = c.next_down(&c.div(&a_lower, &denominator)?)?;
        let upper = c.next_up(&c.div(&a_upper, &denominator)?)?;
        let exact = c.maximum(&self.zero_mask(a)?, &identity)?;
        let lower = c.where_cond(&exact, a.lower, &lower)?;
        let upper = c.where_cond(&exact, a.upper, &upper)?;
        self.bounds(lower, upper)
    }

    /// Outward quotient by intervals with strictly positive lower endpoints.
    pub fn div_positive_interval_outward(
        &self,
        a: IntervalOperand<'_, R>,
        b: IntervalOperand<'_, R>,
    ) -> Result<IntervalPair<R>> {
        let c = self.client;
        let (a_valid, b_valid) = self.div_positive_interval_inputs(a, b)?;
        // Invalid lanes use finite operands before any eager quotient evaluation.
        let a_lower = c.where_cond(&a_valid, a.lower, &self.zero)?;
        let a_upper = c.where_cond(&a_valid, a.upper, &self.zero)?;
        let b_lower = c.where_cond(&b_valid, b.lower, &self.one)?;
        let b_upper = c.where_cond(&b_valid, b.upper, &self.one)?;
        let identity = c.mul(&c.eq(b.lower, &self.one)?, &c.eq(b.upper, &self.one)?)?;
        let a_lower = c.where_cond(&identity, &self.one, &a_lower)?;
        let a_upper = c.where_cond(&identity, &self.one, &a_upper)?;
        let b_lower = c.where_cond(&identity, &self.one, &b_lower)?;
        let b_upper = c.where_cond(&identity, &self.one, &b_upper)?;
        let (lower, upper) = self.corner_extrema(
            c.div(&a_lower, &b_lower)?,
            c.div(&a_lower, &b_upper)?,
            c.div(&a_upper, &b_lower)?,
            c.div(&a_upper, &b_upper)?,
        )?;
        let exact = c.maximum(&self.zero_mask(a)?, &identity)?;
        let lower = c.where_cond(&exact, a.lower, &lower)?;
        let upper = c.where_cond(&exact, a.upper, &upper)?;
        self.bounds(lower, upper)
    }

    /// Mark nonfinite or nonpositive divisors and return the positive-divisor mask.
    pub fn div_positive_inputs(&self, divisor: &Tensor<R>) -> Result<Tensor<R>> {
        self.finite(divisor)?;
        let valid = self.client.gt(divisor, &self.zero)?;
        self.mark(&self.client.eq(&valid, &self.zero)?)?;
        Ok(valid)
    }

    /// Mark invalid numerators and divisors and return both validity masks.
    pub fn div_positive_interval_inputs(
        &self,
        a: IntervalOperand<'_, R>,
        b: IntervalOperand<'_, R>,
    ) -> Result<IntervalPair<R>> {
        let c = self.client;
        let a_valid = self.finite_ordered_mask(a)?;
        let b_valid = c.mul(&self.finite_ordered_mask(b)?, &c.gt(b.lower, &self.zero)?)?;
        self.mark(&c.eq(&a_valid, &self.zero)?)?;
        self.mark(&c.eq(&b_valid, &self.zero)?)?;
        Ok((a_valid, b_valid))
    }

    /// Outward square roots of finite nonnegative intervals.
    pub fn sqrt_nonnegative(&self, a: IntervalOperand<'_, R>) -> Result<IntervalPair<R>> {
        let c = self.client;
        let valid = c.mul(&self.finite_ordered_mask(a)?, &c.ge(a.lower, &self.zero)?)?;
        self.mark(&c.eq(&valid, &self.zero)?)?;
        let lower_input = c.where_cond(&valid, a.lower, &self.zero)?;
        let upper_input = c.where_cond(&valid, a.upper, &self.zero)?;
        let lower = c.next_down(&c.sqrt(&lower_input)?)?;
        let upper = c.next_up(&c.sqrt(&upper_input)?)?;
        let lower = c.where_cond(&c.eq(&lower_input, &self.zero)?, &lower_input, &lower)?;
        let upper = c.where_cond(&c.eq(&upper_input, &self.zero)?, &upper_input, &upper)?;
        self.bounds(lower, upper)
    }

    /// Step the corner minimum down and the corner maximum up.
    ///
    /// Adjacent-value stepping is monotone, so it commutes with these extrema.
    fn corner_extrema(
        &self,
        lower_lower: Tensor<R>,
        lower_upper: Tensor<R>,
        upper_lower: Tensor<R>,
        upper_upper: Tensor<R>,
    ) -> Result<IntervalPair<R>> {
        let c = self.client;
        let minimum = c.minimum(
            &c.minimum(&lower_lower, &lower_upper)?,
            &c.minimum(&upper_lower, &upper_upper)?,
        )?;
        let maximum = c.maximum(
            &c.maximum(&lower_lower, &lower_upper)?,
            &c.maximum(&upper_lower, &upper_upper)?,
        )?;
        Ok((c.next_down(&minimum)?, c.next_up(&maximum)?))
    }
}
