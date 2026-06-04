//! WebGPU implementation of GEMM epilogue operations.

use crate::error::Result;
use crate::ops::{GemmActivation, GemmEpilogueOps};
use crate::runtime::wgpu::ops::native::{
    native_gemm_bias_activation, native_gemm_bias_activation_bwd, native_gemm_bias_residual,
};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

impl GemmEpilogueOps<WgpuRuntime> for WgpuClient {
    fn matmul_bias_activation(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        bias: &Tensor<WgpuRuntime>,
        activation: GemmActivation,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_gemm_bias_activation(self, a, b, bias, activation)
    }

    fn matmul_bias_residual(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        bias: &Tensor<WgpuRuntime>,
        residual: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_gemm_bias_residual(self, a, b, bias, residual)
    }

    fn matmul_bias_activation_bwd(
        &self,
        grad: &Tensor<WgpuRuntime>,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        bias: &Tensor<WgpuRuntime>,
        activation: GemmActivation,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        native_gemm_bias_activation_bwd(self, grad, a, b, bias, activation)
    }
}
