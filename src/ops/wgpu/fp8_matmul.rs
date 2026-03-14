//! WebGPU implementation of FP8 matrix multiplication operations.
//!
//! WebGPU is intentionally limited to 32-bit types (F32, I32, U32).
//! FP8 dtypes are not supported on the WebGPU backend.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::Fp8MatmulOps;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

impl Fp8MatmulOps<WgpuRuntime> for WgpuClient {
    fn fp8_matmul(
        &self,
        a: &Tensor<WgpuRuntime>,
        _b: &Tensor<WgpuRuntime>,
        _scale_a: f32,
        _scale_b: f32,
        _out_dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        Err(Error::UnsupportedDType {
            dtype: a.dtype(),
            op: "fp8_matmul (WebGPU does not support FP8 types)",
        })
    }

    fn fp8_matmul_e5m2(
        &self,
        a: &Tensor<WgpuRuntime>,
        _b: &Tensor<WgpuRuntime>,
        _scale_a: f32,
        _scale_b: f32,
        _out_dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        Err(Error::UnsupportedDType {
            dtype: a.dtype(),
            op: "fp8_matmul_e5m2 (WebGPU does not support FP8 types)",
        })
    }
}
