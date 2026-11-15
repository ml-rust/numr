//! Reference certified expansion operations.

use super::composed::{Composed, ComposedClient};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ExpansionOperand, ExpansionTriple, IntervalOperand, IntervalPair};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

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
    /// Copy `value` into independent storage.
    pub fn snapshot(&self, value: &Tensor<R>) -> Result<Tensor<R>> {
        let copy = Tensor::empty(value.shape(), value.dtype(), self.client.device())?;
        self.client.copy_into(&copy, value)?;
        Ok(copy)
    }

    /// Mark a singleton interval's bounds.
    pub(super) fn point(&self, value: &Tensor<R>) -> Result<()> {
        self.bounds(value.clone(), value.clone()).map(|_| ())
    }

    fn zeros_like(&self, value: &Tensor<R>) -> Result<Tensor<R>> {
        Tensor::zeros(value.shape(), self.dtype, self.client.device())
    }

    /// Mark nonfinite components or a negative radius.
    pub fn expansion_flag(&self, value: ExpansionOperand<'_, R>) -> Result<()> {
        self.finite(value.high)?;
        self.finite(value.low)?;
        self.finite(value.abs_error)?;
        self.mark(&self.client.lt(value.abs_error, &self.zero)?)
    }

    fn owned(
        &self,
        high: Tensor<R>,
        low: Tensor<R>,
        abs_error: Tensor<R>,
    ) -> Result<ExpansionTriple<R>> {
        self.expansion_flag(ExpansionOperand {
            high: &high,
            low: &low,
            abs_error: &abs_error,
        })?;
        Ok((high, low, abs_error))
    }

    /// Exact expansion of represented values.
    pub fn expansion_point(&self, value: &Tensor<R>) -> Result<ExpansionTriple<R>> {
        let high = self.snapshot(value)?;
        let low = self.zeros_like(value)?;
        let abs_error = self.zeros_like(value)?;
        self.owned(high, low, abs_error)
    }

    /// Normalize a supplied center with TwoSum.
    pub fn expansion_normalize(
        &self,
        value: ExpansionOperand<'_, R>,
    ) -> Result<ExpansionTriple<R>> {
        let c = self.client;
        let high = self.snapshot(value.high)?;
        let low = self.snapshot(value.low)?;
        let abs_error = self.snapshot(value.abs_error)?;
        self.finite(&high)?;
        self.finite(&low)?;
        let low_zero = c.eq(&low, &self.zero)?;
        let (normalized_high, normalized_low) = c.two_sum(&high, &low)?;
        let high = c.where_cond(&low_zero, &high, &normalized_high)?;
        let low = c.where_cond(&low_zero, &low, &normalized_low)?;
        self.owned(high, low, abs_error)
    }

    /// Sum or difference of expansion centers.
    pub fn expansion_sum(
        &self,
        a: ExpansionOperand<'_, R>,
        b: ExpansionOperand<'_, R>,
        subtract: bool,
    ) -> Result<ExpansionTriple<R>> {
        self.point(a.abs_error)?;
        self.point(b.abs_error)?;
        let (_, radius) = self.add_outward(point(a.abs_error), point(b.abs_error))?;
        if subtract {
            let b_high = self.client.neg(b.high)?;
            let b_low = self.client.neg(b.low)?;
            self.compress(&[a.high, &b_high, a.low, &b_low], &radius)
        } else {
            self.compress(&[a.high, b.high, a.low, b.low], &radius)
        }
    }

    /// Compress terms with TwoSum while bounding every discarded residual.
    pub(super) fn compress(
        &self,
        terms: &[&Tensor<R>],
        abs_error: &Tensor<R>,
    ) -> Result<ExpansionTriple<R>> {
        let c = self.client;
        let Some((first, rest)) = terms.split_first() else {
            return Err(crate::error::Error::InvalidArgument {
                arg: "expansion",
                reason: "Expansion compression requires a center term. Supply at least one term."
                    .into(),
            });
        };
        for term in rest {
            self.finite(term)?;
        }
        self.finite(first)?;
        self.finite(abs_error)?;
        let mut high = self.snapshot(first)?;
        let mut low = self.zeros_like(first)?;
        let mut radius = self.snapshot(abs_error)?;
        self.mark(&c.lt(abs_error, &self.zero)?)?;
        for term in rest {
            let (sum, residual) = c.two_sum(&high, term)?;
            let (low_sum, discarded) = c.two_sum(&low, &residual)?;
            let (next_high, next_low) = c.two_sum(&sum, &low_sum)?;
            self.finite(&discarded)?;
            let discarded = c.abs(&discarded)?;
            self.point(&radius)?;
            self.point(&discarded)?;
            radius = self.add_outward(point(&radius), point(&discarded))?.1;
            high = next_high;
            low = next_low;
        }
        self.owned(high, low, radius)
    }

    /// Project an expansion into an outward interval.
    pub fn expansion_interval(&self, value: ExpansionOperand<'_, R>) -> Result<IntervalPair<R>> {
        self.point(value.high)?;
        self.point(value.low)?;
        let (center_lower, center_upper) = self.add_outward(point(value.high), point(value.low))?;
        let center = IntervalOperand {
            lower: &center_lower,
            upper: &center_upper,
        };
        self.point(value.abs_error)?;
        let (lower, _) = self.sub_outward(center, point(value.abs_error))?;
        let (_, upper) = self.add_outward(center, point(value.abs_error))?;
        self.bounds(lower, upper)
    }
}
