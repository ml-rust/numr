//! WGPU expansion operations through the composed reference. F64 is unsupported.

use super::interval::f32_only;
use crate::error::Result;
use crate::ops::impl_generic::interval::{
    ExpansionBinary, expansion_binary_impl, expansion_flag_impl, expansion_from_interval_impl,
    expansion_interval_impl, expansion_normalize_impl, expansion_point_impl,
};
use crate::ops::{ExpansionOperand, ExpansionOps, ExpansionTriple, IntervalOperand, IntervalPair};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

type WgpuTensor = Tensor<WgpuRuntime>;

fn components<'a>(value: ExpansionOperand<'a, WgpuRuntime>) -> [&'a WgpuTensor; 3] {
    [value.high, value.low, value.abs_error]
}

fn binary(
    client: &WgpuClient,
    op: ExpansionBinary,
    a: ExpansionOperand<'_, WgpuRuntime>,
    b: ExpansionOperand<'_, WgpuRuntime>,
    status: &WgpuTensor,
) -> Result<ExpansionTriple<WgpuRuntime>> {
    f32_only(&components(a), op.name())?;
    f32_only(&components(b), op.name())?;
    expansion_binary_impl(client, op, a, b, status)
}

impl ExpansionOps<WgpuRuntime> for WgpuClient {
    fn expansion_point(
        &self,
        value: &WgpuTensor,
        status: &WgpuTensor,
    ) -> Result<ExpansionTriple<WgpuRuntime>> {
        f32_only(&[value], "expansion_point")?;
        expansion_point_impl(self, value, status)
    }

    fn expansion_normalize(
        &self,
        value: ExpansionOperand<'_, WgpuRuntime>,
        status: &WgpuTensor,
    ) -> Result<ExpansionTriple<WgpuRuntime>> {
        f32_only(&components(value), "expansion_normalize")?;
        expansion_normalize_impl(self, value, status)
    }

    fn expansion_flag(
        &self,
        value: ExpansionOperand<'_, WgpuRuntime>,
        status: &WgpuTensor,
    ) -> Result<()> {
        f32_only(&components(value), "expansion_flag")?;
        expansion_flag_impl(self, value, status)
    }

    fn expansion_add(
        &self,
        a: ExpansionOperand<'_, WgpuRuntime>,
        b: ExpansionOperand<'_, WgpuRuntime>,
        status: &WgpuTensor,
    ) -> Result<ExpansionTriple<WgpuRuntime>> {
        binary(self, ExpansionBinary::Add, a, b, status)
    }

    fn expansion_sub(
        &self,
        a: ExpansionOperand<'_, WgpuRuntime>,
        b: ExpansionOperand<'_, WgpuRuntime>,
        status: &WgpuTensor,
    ) -> Result<ExpansionTriple<WgpuRuntime>> {
        binary(self, ExpansionBinary::Sub, a, b, status)
    }

    fn expansion_mul(
        &self,
        a: ExpansionOperand<'_, WgpuRuntime>,
        b: ExpansionOperand<'_, WgpuRuntime>,
        status: &WgpuTensor,
    ) -> Result<ExpansionTriple<WgpuRuntime>> {
        binary(self, ExpansionBinary::Mul, a, b, status)
    }

    fn expansion_div_positive(
        &self,
        a: ExpansionOperand<'_, WgpuRuntime>,
        b: ExpansionOperand<'_, WgpuRuntime>,
        status: &WgpuTensor,
    ) -> Result<ExpansionTriple<WgpuRuntime>> {
        binary(self, ExpansionBinary::DivPositive, a, b, status)
    }

    fn expansion_from_interval(
        &self,
        value: IntervalOperand<'_, WgpuRuntime>,
        status: &WgpuTensor,
    ) -> Result<ExpansionTriple<WgpuRuntime>> {
        f32_only(&[value.lower, value.upper], "expansion_from_interval")?;
        expansion_from_interval_impl(self, value, status)
    }

    fn expansion_interval(
        &self,
        value: ExpansionOperand<'_, WgpuRuntime>,
        status: &WgpuTensor,
    ) -> Result<IntervalPair<WgpuRuntime>> {
        f32_only(&components(value), "expansion_interval")?;
        expansion_interval_impl(self, value, status)
    }
}
