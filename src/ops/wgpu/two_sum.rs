//! WGPU error-free addition through configured tensor primitives.

use crate::error::Result;
use crate::ops::TwoSumOps;
use crate::ops::impl_generic::two_sum::two_sum_impl;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

impl TwoSumOps<WgpuRuntime> for WgpuClient {
    fn two_sum(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        if a.dtype() == crate::dtype::DType::F64 {
            return Err(crate::error::Error::UnsupportedDType {
                dtype: a.dtype(),
                op: "two_sum",
            });
        }
        two_sum_impl(self, a, b)
    }
}
