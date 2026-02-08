//! ScalarOps trait implementation for WebGPU runtime.

use super::super::{WgpuClient, WgpuRuntime};
use super::native::*;
use crate::error::Result;
use crate::ops::ScalarOps;
use crate::tensor::Tensor;

impl ScalarOps<WgpuRuntime> for WgpuClient {
    fn add_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "add_scalar", a, scalar)
    }

    fn sub_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "sub_scalar", a, scalar)
    }

    fn mul_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "mul_scalar", a, scalar)
    }

    fn div_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "div_scalar", a, scalar)
    }

    fn pow_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "pow_scalar", a, scalar)
    }

    fn rsub_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "rsub_scalar", a, scalar)
    }
}
