//! Reference interval rules composed from ordinary rounded tensor operations.
//!
//! Native kernels reproduce these compositions bit for bit.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{
    BinaryOps, CompareOps, ConditionalOps, IntervalOperand, IntervalPair, ReduceOps,
    TypeConversionOps, UnaryOps,
};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Primitive operations the reference compositions use.
pub trait ComposedClient<R: Runtime<DType = DType>>:
    BinaryOps<R>
    + CompareOps<R>
    + ConditionalOps<R>
    + ReduceOps<R>
    + TypeConversionOps<R>
    + UnaryOps<R>
    + crate::ops::MidpointOps<R>
    + crate::ops::NextRepresentableOps<R>
    + crate::ops::TwoSumOps<R>
    + RuntimeClient<R>
{
}

impl<R, C> ComposedClient<R> for C
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + crate::ops::MidpointOps<R>
        + crate::ops::NextRepresentableOps<R>
        + crate::ops::TwoSumOps<R>
        + RuntimeClient<R>,
{
}

/// Client, dtype constants, and status flag for one reference operation.
pub struct Composed<'a, R: Runtime<DType = DType>, C> {
    pub(super) client: &'a C,
    pub(super) dtype: DType,
    pub(super) zero: Tensor<R>,
    pub(super) one: Tensor<R>,
    pub(super) status: &'a Tensor<R>,
}

impl<'a, R, C> Composed<'a, R, C>
where
    R: Runtime<DType = DType>,
    C: ComposedClient<R>,
{
    /// Scalar constants on the client's device.
    pub fn new(client: &'a C, dtype: DType, status: &'a Tensor<R>) -> Result<Self> {
        Ok(Self {
            client,
            dtype,
            zero: Tensor::zeros(&[], dtype, client.device())?,
            one: Tensor::ones(&[], dtype, client.device())?,
            status,
        })
    }

    /// Whether exact singleton retention through fused residuals applies.
    pub(super) fn fused(&self) -> bool {
        R::fused_mul_add_contract(self.dtype) == crate::runtime::FusedMulAddContract::Ieee754Fused
    }

    /// Merge `any(mask)` into the status flag.
    pub fn mark(&self, mask: &Tensor<R>) -> Result<()> {
        if mask.numel() == 0 {
            return Ok(());
        }
        let any = self.client.any(mask, &[], false)?;
        let any = self.client.cast(&any, DType::U32)?;
        let merged = self.client.maximum(self.status, &any)?;
        self.client.copy_into(self.status, &merged)
    }

    /// Mark NaN or infinite elements.
    pub fn finite(&self, value: &Tensor<R>) -> Result<()> {
        let nan = self.client.cast(&self.client.isnan(value)?, self.dtype)?;
        let infinite = self.client.cast(&self.client.isinf(value)?, self.dtype)?;
        self.mark(&self.client.maximum(&nan, &infinite)?)
    }

    /// Mark nonfinite or inverted bounds, then return them.
    pub fn bounds(&self, lower: Tensor<R>, upper: Tensor<R>) -> Result<IntervalPair<R>> {
        self.finite(&lower)?;
        self.finite(&upper)?;
        self.mark(&self.client.gt(&lower, &upper)?)?;
        Ok((lower, upper))
    }

    /// Lanes where both endpoints equal zero.
    pub(super) fn zero_mask(&self, value: IntervalOperand<'_, R>) -> Result<Tensor<R>> {
        self.client.mul(
            &self.client.eq(value.lower, &self.zero)?,
            &self.client.eq(value.upper, &self.zero)?,
        )
    }

    /// Lanes with finite ordered endpoints.
    pub(super) fn finite_ordered_mask(&self, value: IntervalOperand<'_, R>) -> Result<Tensor<R>> {
        let c = self.client;
        let lower_nan = c.cast(&c.isnan(value.lower)?, self.dtype)?;
        let lower_infinite = c.cast(&c.isinf(value.lower)?, self.dtype)?;
        let upper_nan = c.cast(&c.isnan(value.upper)?, self.dtype)?;
        let upper_infinite = c.cast(&c.isinf(value.upper)?, self.dtype)?;
        let nonfinite = c.maximum(
            &c.maximum(&lower_nan, &lower_infinite)?,
            &c.maximum(&upper_nan, &upper_infinite)?,
        )?;
        c.mul(
            &c.eq(&nonfinite, &self.zero)?,
            &c.le(value.lower, value.upper)?,
        )
    }

    /// Lanes where `value` is finite.
    pub(super) fn finite_mask(&self, value: &Tensor<R>) -> Result<Tensor<R>> {
        let c = self.client;
        let nan = c.cast(&c.isnan(value)?, self.dtype)?;
        let infinite = c.cast(&c.isinf(value)?, self.dtype)?;
        c.eq(&c.maximum(&nan, &infinite)?, &self.zero)
    }

    /// Magnitude limits that keep guarded residuals exact.
    ///
    /// Guarded F64 products lie on the 2^-1004 lattice. F32 products use 2^-126.
    /// Nonzero residuals cannot disappear through gradual underflow on these lattices.
    pub(super) fn limits(&self) -> Result<(Tensor<R>, Tensor<R>)> {
        let (minimum, maximum) = match self.dtype {
            DType::F32 => (2f64.powi(-40), 2f64.powi(40)),
            _ => (2f64.powi(-450), 2f64.powi(450)),
        };
        Ok((
            Tensor::full_scalar(&[], self.dtype, minimum, self.client.device())?,
            Tensor::full_scalar(&[], self.dtype, maximum, self.client.device())?,
        ))
    }

    /// Lanes with magnitude inside the guarded limits.
    pub(super) fn nonzero_range(
        &self,
        value: &Tensor<R>,
        limits: &(Tensor<R>, Tensor<R>),
    ) -> Result<Tensor<R>> {
        let magnitude = self.client.abs(value)?;
        self.client.mul(
            &self.client.ge(&magnitude, &limits.0)?,
            &self.client.le(&magnitude, &limits.1)?,
        )
    }
}
