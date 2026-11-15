//! WGPU interval operations through the composed reference. F64 is unsupported.
//!
//! WGPU declares no IEEE arithmetic contract, so certified callers reject it at admission.
//! The composed reference keeps every operation available on F32 tensors.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::impl_generic::interval::envelope::interval_envelope_impl;
use crate::ops::impl_generic::interval::{
    IntervalBinary, interval_binary_impl, interval_capture_impl, interval_flag_bounds_impl,
    interval_flag_nonfinite_impl, interval_flag_nonzero_impl, interval_sqrt_nonnegative_impl,
    interval_status_impl,
};
use crate::ops::{IntervalEnclosure, IntervalOperand, IntervalOps, IntervalPair};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

type WgpuTensor = Tensor<WgpuRuntime>;

/// Refuse F64 operands, which WGSL cannot compute.
pub(super) fn f32_only(values: &[&WgpuTensor], op: &'static str) -> Result<()> {
    match values.iter().find(|value| value.dtype() == DType::F64) {
        Some(value) => Err(Error::UnsupportedDType {
            dtype: value.dtype(),
            op,
        }),
        None => Ok(()),
    }
}

fn binary(
    client: &WgpuClient,
    op: IntervalBinary,
    a: IntervalOperand<'_, WgpuRuntime>,
    b: IntervalOperand<'_, WgpuRuntime>,
    enclosure: IntervalEnclosure,
    status: &WgpuTensor,
) -> Result<IntervalPair<WgpuRuntime>> {
    f32_only(&[a.lower, a.upper, b.lower, b.upper], op.name())?;
    interval_binary_impl(client, op, a, b, enclosure, status)
}

impl IntervalOps<WgpuRuntime> for WgpuClient {
    fn interval_status(&self) -> Result<WgpuTensor> {
        interval_status_impl(self)
    }

    fn interval_envelope(&self, dtype: DType) -> Result<WgpuTensor> {
        if dtype == DType::F64 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "interval_envelope",
            });
        }
        interval_envelope_impl(self, dtype)
    }

    fn interval_flag_nonzero(&self, mask: &WgpuTensor, status: &WgpuTensor) -> Result<()> {
        f32_only(&[mask], "interval_flag_nonzero")?;
        interval_flag_nonzero_impl(self, mask, status)
    }

    fn interval_flag_nonfinite(&self, value: &WgpuTensor, status: &WgpuTensor) -> Result<()> {
        f32_only(&[value], "interval_flag_nonfinite")?;
        interval_flag_nonfinite_impl(self, value, status)
    }

    fn interval_flag_bounds(
        &self,
        value: IntervalOperand<'_, WgpuRuntime>,
        status: &WgpuTensor,
    ) -> Result<()> {
        f32_only(&[value.lower, value.upper], "interval_flag_bounds")?;
        interval_flag_bounds_impl(self, value, status)
    }

    fn interval_capture(
        &self,
        value: IntervalOperand<'_, WgpuRuntime>,
        status: &WgpuTensor,
    ) -> Result<IntervalPair<WgpuRuntime>> {
        f32_only(&[value.lower, value.upper], "interval_capture")?;
        interval_capture_impl(self, value, status)
    }

    fn interval_add(
        &self,
        a: IntervalOperand<'_, WgpuRuntime>,
        b: IntervalOperand<'_, WgpuRuntime>,
        enclosure: IntervalEnclosure,
        status: &WgpuTensor,
    ) -> Result<IntervalPair<WgpuRuntime>> {
        binary(self, IntervalBinary::Add, a, b, enclosure, status)
    }

    fn interval_sub(
        &self,
        a: IntervalOperand<'_, WgpuRuntime>,
        b: IntervalOperand<'_, WgpuRuntime>,
        enclosure: IntervalEnclosure,
        status: &WgpuTensor,
    ) -> Result<IntervalPair<WgpuRuntime>> {
        binary(self, IntervalBinary::Sub, a, b, enclosure, status)
    }

    fn interval_mul(
        &self,
        a: IntervalOperand<'_, WgpuRuntime>,
        b: IntervalOperand<'_, WgpuRuntime>,
        enclosure: IntervalEnclosure,
        status: &WgpuTensor,
    ) -> Result<IntervalPair<WgpuRuntime>> {
        binary(self, IntervalBinary::Mul, a, b, enclosure, status)
    }

    fn interval_div_positive(
        &self,
        a: IntervalOperand<'_, WgpuRuntime>,
        divisor: &WgpuTensor,
        enclosure: IntervalEnclosure,
        status: &WgpuTensor,
    ) -> Result<IntervalPair<WgpuRuntime>> {
        let b = IntervalOperand {
            lower: divisor,
            upper: divisor,
        };
        binary(self, IntervalBinary::DivPositive, a, b, enclosure, status)
    }

    fn interval_div_positive_interval(
        &self,
        a: IntervalOperand<'_, WgpuRuntime>,
        b: IntervalOperand<'_, WgpuRuntime>,
        enclosure: IntervalEnclosure,
        status: &WgpuTensor,
    ) -> Result<IntervalPair<WgpuRuntime>> {
        binary(
            self,
            IntervalBinary::DivPositiveInterval,
            a,
            b,
            enclosure,
            status,
        )
    }

    fn interval_sqrt_nonnegative(
        &self,
        a: IntervalOperand<'_, WgpuRuntime>,
        status: &WgpuTensor,
    ) -> Result<IntervalPair<WgpuRuntime>> {
        f32_only(&[a.lower, a.upper], "interval_sqrt_nonnegative")?;
        interval_sqrt_nonnegative_impl(self, a, status)
    }
}
