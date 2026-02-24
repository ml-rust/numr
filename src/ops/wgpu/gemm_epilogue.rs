//! WebGPU implementation of GEMM epilogue operations.

use crate::error::{Error, Result};
use crate::ops::{GemmActivation, GemmEpilogueOps};
use crate::runtime::wgpu::ops::native::{native_gemm_bias_activation, native_gemm_bias_residual};
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
        _grad: &Tensor<WgpuRuntime>,
        _a: &Tensor<WgpuRuntime>,
        _b: &Tensor<WgpuRuntime>,
        _bias: &Tensor<WgpuRuntime>,
        _activation: GemmActivation,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        Err(Error::NotImplemented {
            feature: "matmul_bias_activation_bwd on WebGPU; use CPU backend for training",
        })
    }
}
