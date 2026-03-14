//! Binary operations for WebGPU runtime

use crate::error::Result;
use crate::ops::BinaryOps;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::native::{
    native_binary_op, native_fused_add_mul, native_fused_mul_add,
};
use crate::tensor::Tensor;

impl BinaryOps<WgpuRuntime> for WgpuClient {
    fn add(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "add", a, b)
    }

    fn sub(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "sub", a, b)
    }

    fn mul(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "mul", a, b)
    }

    fn div(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "div", a, b)
    }

    fn pow(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "pow", a, b)
    }

    fn maximum(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "maximum", a, b)
    }

    fn minimum(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "minimum", a, b)
    }

    fn atan2(
        &self,
        y: &Tensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_binary_op(self, "atan2", y, x)
    }

    fn fused_mul_add(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        c: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_fused_mul_add(self, a, b, c)
    }

    fn fused_add_mul(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        c: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_fused_add_mul(self, a, b, c)
    }
}
