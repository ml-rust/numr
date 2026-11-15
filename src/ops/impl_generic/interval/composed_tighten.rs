//! Reference singleton retention and sign restriction for outward results.

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
    /// Retain singleton sums or differences that TwoSum proves exact.
    pub fn tighten_sum(
        &self,
        a: IntervalOperand<'_, R>,
        b: IntervalOperand<'_, R>,
        outward: IntervalOperand<'_, R>,
        subtract: bool,
    ) -> Result<IntervalPair<R>> {
        let c = self.client;
        let (_, maximum) = self.limits()?;
        let zero_identity = c.maximum(&self.zero_mask(a)?, &self.zero_mask(b)?)?;
        let valid = c.mul(
            &self.finite_ordered_mask(outward)?,
            &c.eq(&zero_identity, &self.zero)?,
        )?;
        let (b_lower, b_upper) = if subtract {
            (c.neg(b.upper)?, c.neg(b.lower)?)
        } else {
            (b.lower.clone(), b.upper.clone())
        };
        let lower = self.exact_sum(a.lower, &b_lower, outward.lower, &valid, &maximum)?;
        let upper = self.exact_sum(a.upper, &b_upper, outward.upper, &valid, &maximum)?;
        self.bounds(lower, upper)
    }

    fn exact_sum(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
        outward: &Tensor<R>,
        valid: &Tensor<R>,
        maximum: &Tensor<R>,
    ) -> Result<Tensor<R>> {
        let c = self.client;
        // The upper guard prevents overflow inside TwoSum. Gradual underflow preserves additive residuals.
        let candidate = c.mul(
            valid,
            &c.mul(&c.le(&c.abs(a)?, maximum)?, &c.le(&c.abs(b)?, maximum)?)?,
        )?;
        let safe_a = c.where_cond(&candidate, a, &self.zero)?;
        let safe_b = c.where_cond(&candidate, b, &self.zero)?;
        let (sum, residual) = c.two_sum(&safe_a, &safe_b)?;
        let exact = c.mul(&candidate, &c.eq(&residual, &self.zero)?)?;
        c.where_cond(&exact, &sum, outward)
    }

    /// Retain singleton products that a fused residual proves exact.
    pub fn tighten_mul(
        &self,
        a: IntervalOperand<'_, R>,
        b: IntervalOperand<'_, R>,
        outward: IntervalOperand<'_, R>,
    ) -> Result<IntervalPair<R>> {
        let c = self.client;
        let candidate = c.mul(
            &c.mul(&c.eq(a.lower, a.upper)?, &c.eq(b.lower, b.upper)?)?,
            &c.mul(
                &c.mul(&self.finite_ordered_mask(a)?, &self.finite_ordered_mask(b)?)?,
                &self.finite_ordered_mask(outward)?,
            )?,
        )?;
        let zero_identity = c.maximum(&self.zero_mask(a)?, &self.zero_mask(b)?)?;
        let candidate = c.mul(&candidate, &c.eq(&zero_identity, &self.zero)?)?;
        let safe_a = c.where_cond(&candidate, a.lower, &self.one)?;
        let safe_b = c.where_cond(&candidate, b.lower, &self.one)?;
        let product = c.mul(&safe_a, &safe_b)?;
        let candidate = c.mul(
            &candidate,
            &self.product_residual_exact_mask(&safe_a, &safe_b, &product)?,
        )?;
        // Guard the residual separately: a rounded zero can conceal a nonzero exact product.
        let safe_a = c.where_cond(&candidate, &safe_a, &self.one)?;
        let safe_b = c.where_cond(&candidate, &safe_b, &self.one)?;
        let safe_product = c.where_cond(&candidate, &product, &self.one)?;
        let shape = outward.lower.shape();
        let residual = c.fused_mul_add(
            &safe_a.broadcast_to(shape)?,
            &safe_b.broadcast_to(shape)?,
            &c.neg(&safe_product)?.broadcast_to(shape)?,
        )?;
        self.replace_exact(outward, &candidate, &product, &residual)
    }

    /// Retain singleton quotients that a fused residual proves exact.
    pub fn tighten_div(
        &self,
        a: IntervalOperand<'_, R>,
        b: IntervalOperand<'_, R>,
        outward: IntervalOperand<'_, R>,
    ) -> Result<IntervalPair<R>> {
        let c = self.client;
        let limits = self.limits()?;
        let a_candidate = c.mul(
            &c.eq(a.lower, a.upper)?,
            &self.nonzero_range(a.lower, &limits)?,
        )?;
        let b_candidate = c.mul(
            &c.eq(b.lower, b.upper)?,
            &self.nonzero_range(b.lower, &limits)?,
        )?;
        let singleton = c.mul(
            &c.mul(&a_candidate, &b_candidate)?,
            &self.finite_ordered_mask(outward)?,
        )?;
        let candidate = c.mul(&singleton, &c.gt(b.lower, &self.zero)?)?;
        let safe_a = c.where_cond(&candidate, a.lower, &self.one)?;
        let safe_b = c.where_cond(&candidate, b.lower, &self.one)?;
        let quotient = c.div(&safe_a, &safe_b)?;
        let candidate = c.mul(&candidate, &self.nonzero_range(&quotient, &limits)?)?;
        let safe_a = c.where_cond(&candidate, &safe_a, &self.one)?;
        let safe_b = c.where_cond(&candidate, &safe_b, &self.one)?;
        let safe_quotient = c.where_cond(&candidate, &quotient, &self.one)?;
        let shape = outward.lower.shape();
        let residual = c.fused_mul_add(
            &c.neg(&safe_quotient)?.broadcast_to(shape)?,
            &safe_b.broadcast_to(shape)?,
            &safe_a.broadcast_to(shape)?,
        )?;
        self.replace_exact(outward, &candidate, &quotient, &residual)
    }

    /// Lanes whose fused product residual is exact.
    pub fn product_residual_exact_mask(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
        product: &Tensor<R>,
    ) -> Result<Tensor<R>> {
        let c = self.client;
        let limits = self.limits()?;
        let guarded = c.mul(
            &c.mul(
                &self.nonzero_range(a, &limits)?,
                &self.nonzero_range(b, &limits)?,
            )?,
            &self.nonzero_range(product, &limits)?,
        )?;
        let zero_operand = c.maximum(&c.eq(a, &self.zero)?, &c.eq(b, &self.zero)?)?;
        let finite_a = self.finite_mask(a)?;
        let finite_b = self.finite_mask(b)?;
        let safe_a = c.where_cond(&finite_a, a, &self.zero)?;
        let safe_b = c.where_cond(&finite_b, b, &self.zero)?;
        let integer_factor = c.maximum(
            &c.eq(&safe_a, &c.trunc(&safe_a)?)?,
            &c.eq(&safe_b, &c.trunc(&safe_b)?)?,
        )?;
        // An integer factor keeps the exact product on the minimum-subnormal lattice.
        // A nonzero fused residual therefore cannot disappear through gradual underflow.
        let proof = c.maximum(&c.maximum(&guarded, &zero_operand)?, &integer_factor)?;
        c.mul(
            &proof,
            &c.mul(&c.mul(&finite_a, &finite_b)?, &self.finite_mask(product)?)?,
        )
    }

    fn replace_exact(
        &self,
        outward: IntervalOperand<'_, R>,
        candidate: &Tensor<R>,
        nominal: &Tensor<R>,
        residual: &Tensor<R>,
    ) -> Result<IntervalPair<R>> {
        let c = self.client;
        let exact = c.mul(candidate, &c.eq(residual, &self.zero)?)?;
        let lower = c.where_cond(&exact, nominal, outward.lower)?;
        let upper = c.where_cond(&exact, nominal, outward.upper)?;
        self.bounds(lower, upper)
    }

    /// Clamp product bounds to signs that operand endpoints prove.
    pub fn restrict_mul_sign(
        &self,
        a: IntervalOperand<'_, R>,
        b: IntervalOperand<'_, R>,
        outward: IntervalOperand<'_, R>,
    ) -> Result<IntervalPair<R>> {
        let c = self.client;
        let a_nonnegative = c.ge(a.lower, &self.zero)?;
        let a_nonpositive = c.le(a.upper, &self.zero)?;
        let b_nonnegative = c.ge(b.lower, &self.zero)?;
        let b_nonpositive = c.le(b.upper, &self.zero)?;
        let nonnegative = c.maximum(
            &c.mul(&a_nonnegative, &b_nonnegative)?,
            &c.mul(&a_nonpositive, &b_nonpositive)?,
        )?;
        let nonpositive = c.maximum(
            &c.mul(&a_nonnegative, &b_nonpositive)?,
            &c.mul(&a_nonpositive, &b_nonnegative)?,
        )?;
        self.restrict_sign(outward, &nonnegative, &nonpositive)
    }

    /// Clamp quotient bounds to the numerator's proved sign.
    pub fn restrict_div_sign(
        &self,
        a: IntervalOperand<'_, R>,
        outward: IntervalOperand<'_, R>,
    ) -> Result<IntervalPair<R>> {
        let nonnegative = self.client.ge(a.lower, &self.zero)?;
        let nonpositive = self.client.le(a.upper, &self.zero)?;
        self.restrict_sign(outward, &nonnegative, &nonpositive)
    }

    fn restrict_sign(
        &self,
        outward: IntervalOperand<'_, R>,
        nonnegative: &Tensor<R>,
        nonpositive: &Tensor<R>,
    ) -> Result<IntervalPair<R>> {
        let c = self.client;
        // Strict comparisons preserve represented zero endpoints, including their sign bits.
        let replace_lower = c.mul(nonnegative, &c.lt(outward.lower, &self.zero)?)?;
        let replace_upper = c.mul(nonpositive, &c.gt(outward.upper, &self.zero)?)?;
        let lower = c.where_cond(&replace_lower, &self.zero, outward.lower)?;
        let upper = c.where_cond(&replace_upper, &self.zero, outward.upper)?;
        self.bounds(lower, upper)
    }
}
