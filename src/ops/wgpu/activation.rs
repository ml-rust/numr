//! Activation operations for WebGPU runtime

use crate::error::Result;
use crate::ops::ActivationOps;
use crate::ops::impl_generic::activation::{dropout_impl, log_softmax_impl, softplus_impl};
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::native::{
    native_parametric_activation, native_softmax, native_unary_op,
};
use crate::tensor::Tensor;

impl ActivationOps<WgpuRuntime> for WgpuClient {
    fn relu(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "relu", a)
    }

    fn sigmoid(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "sigmoid", a)
    }

    fn softmax(&self, a: &Tensor<WgpuRuntime>, dim: isize) -> Result<Tensor<WgpuRuntime>> {
        native_softmax(self, a, dim)
    }

    fn silu(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "silu", a)
    }

    fn gelu(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "gelu", a)
    }

    fn leaky_relu(
        &self,
        a: &Tensor<WgpuRuntime>,
        negative_slope: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_parametric_activation(self, "leaky_relu", a, negative_slope)
    }

    fn elu(&self, a: &Tensor<WgpuRuntime>, alpha: f64) -> Result<Tensor<WgpuRuntime>> {
        native_parametric_activation(self, "elu", a, alpha)
    }

    fn softplus(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        softplus_impl(self, a)
    }

    fn log_softmax(&self, a: &Tensor<WgpuRuntime>, dim: isize) -> Result<Tensor<WgpuRuntime>> {
        log_softmax_impl(self, a, dim)
    }

    fn dropout(
        &self,
        a: &Tensor<WgpuRuntime>,
        p: f64,
        training: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        dropout_impl(self, a, p, training)
    }
}
