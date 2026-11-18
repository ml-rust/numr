//! WGPU rejection of the correctly rounded F64 functions.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::CorrectlyRoundedOps;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

impl CorrectlyRoundedOps<WgpuRuntime> for WgpuClient {
    fn cr_exp(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        reject(a, "cr_exp")
    }

    fn cr_log(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        reject(a, "cr_log")
    }

    fn cr_log1p(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        reject(a, "cr_log1p")
    }

    fn cr_pow_scalar(
        &self,
        a: &Tensor<WgpuRuntime>,
        _exponent: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        reject(a, "cr_pow_scalar")
    }
}

/// The family is F64 only, and WebGPU computes no F64.
fn reject(a: &Tensor<WgpuRuntime>, op: &'static str) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    if dtype != DType::F64 {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Err(Error::backend_limitation(
        "wgpu",
        op,
        "dtype=F64, WebGPU computes no F64. Run this op on a runtime that implements CorrectlyRoundedOps for F64, such as CPU",
    ))
}
