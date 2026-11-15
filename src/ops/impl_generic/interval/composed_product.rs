//! Reference expansion product with fused residual enclosures.

use super::composed::{Composed, ComposedClient};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ExpansionOperand, ExpansionTriple, IntervalOperand};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

impl<R, C> Composed<'_, R, C>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    /// Enclose all four center products and both input radii.
    pub fn expansion_mul(
        &self,
        a: ExpansionOperand<'_, R>,
        b: ExpansionOperand<'_, R>,
    ) -> Result<ExpansionTriple<R>> {
        let c = self.client;
        let exact_one = |value: ExpansionOperand<'_, R>| -> Result<Tensor<R>> {
            c.mul(
                &c.eq(value.high, &self.one)?,
                &c.mul(
                    &c.eq(value.low, &self.zero)?,
                    &c.eq(value.abs_error, &self.zero)?,
                )?,
            )
        };
        let a_one = exact_one(a)?;
        let b_one = exact_one(b)?;
        let identity = c.maximum(&a_one, &b_one)?;
        let mask = |value: ExpansionOperand<'_, R>| -> Result<ExpansionTriple<R>> {
            Ok((
                c.where_cond(&identity, &self.one, value.high)?,
                c.where_cond(&identity, &self.zero, value.low)?,
                c.where_cond(&identity, &self.zero, value.abs_error)?,
            ))
        };
        let (ah, al, ae) = mask(a)?;
        let (bh, bl, be) = mask(b)?;

        let (hh, hh_low, hh_error) = self.component_product(&ah, &bh)?;
        let (hl, hl_low, hl_error) = self.component_product(&ah, &bl)?;
        let (lh, lh_low, lh_error) = self.component_product(&al, &bh)?;
        let (ll, ll_low, ll_error) = self.component_product(&al, &bl)?;

        let terms = [
            self.radius_product(&ah, &be)?,
            self.radius_product(&al, &be)?,
            self.radius_product(&bh, &ae)?,
            self.radius_product(&bl, &ae)?,
            self.radius_product(&ae, &be)?,
            hh_error,
            hl_error,
            lh_error,
            ll_error,
        ];
        for term in &terms {
            self.point(term)?;
        }
        let mut radius = (terms[0].clone(), terms[0].clone());
        for term in &terms[1..] {
            radius = self.add_outward(
                IntervalOperand {
                    lower: &radius.0,
                    upper: &radius.1,
                },
                IntervalOperand {
                    lower: term,
                    upper: term,
                },
            )?;
        }
        let (high, low, abs_error) = self.compress(
            &[&hh, &hh_low, &hl, &hl_low, &lh, &lh_low, &ll, &ll_low],
            &radius.1,
        )?;
        let select = |a: &Tensor<R>, b: &Tensor<R>, general: &Tensor<R>| -> Result<Tensor<R>> {
            c.where_cond(&identity, &c.where_cond(&a_one, b, a)?, general)
        };
        let high = select(a.high, b.high, &high)?;
        let low = select(a.low, b.low, &low)?;
        let abs_error = select(a.abs_error, b.abs_error, &abs_error)?;
        self.expansion_flag(ExpansionOperand {
            high: &high,
            low: &low,
            abs_error: &abs_error,
        })?;
        Ok((high, low, abs_error))
    }

    fn component_product(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<ExpansionTriple<R>> {
        let c = self.client;
        let product = c.mul(a, b)?;
        self.finite(&product)?;
        let residual = c.fused_mul_add(a, b, &c.neg(&product)?)?;
        self.finite(&residual)?;
        let exact = self.product_residual_exact_mask(a, b, &product)?;
        // Adjacent differences are exact, including gaps around zero and subnormal residuals.
        let lower_gap = c.sub(&residual, &c.next_down(&residual)?)?;
        let upper_gap = c.sub(&c.next_up(&residual)?, &residual)?;
        let gap = c.maximum(&lower_gap, &upper_gap)?;
        self.finite(&gap)?;
        let zero = Tensor::zeros(a.shape(), a.dtype(), c.device())?;
        let error = c.where_cond(&exact, &zero, &gap)?;
        Ok((product, residual, error))
    }

    fn radius_product(&self, component: &Tensor<R>, radius: &Tensor<R>) -> Result<Tensor<R>> {
        let magnitude = self.client.abs(component)?;
        self.point(&magnitude)?;
        self.point(radius)?;
        let operand = |value| IntervalOperand {
            lower: value,
            upper: value,
        };
        Ok(self.mul_outward(operand(&magnitude), operand(radius))?.1)
    }
}
