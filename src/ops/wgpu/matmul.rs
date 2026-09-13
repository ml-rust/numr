//! Matrix multiplication operations for WebGPU runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{MatmulOps, TypeConversionOps};
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::native::{native_matmul, native_matmul_bias};
use crate::tensor::Tensor;

impl MatmulOps<WgpuRuntime> for WgpuClient {
    fn matmul(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_matmul(self, a, b)
    }

    /// WebGPU has no half GEMM that keeps its F32 accumulator, so a half
    /// operand is cast to F32 and multiplied there: the same product, one
    /// rounding per operand element on the way in and none on the way out.
    /// Every other dtype is `matmul`.
    fn matmul_wide(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        if !matches!(a.dtype(), DType::F16 | DType::BF16) {
            return native_matmul(self, a, b);
        }
        let a32 = self.cast(a, DType::F32)?;
        let b32 = self.cast(b, DType::F32)?;
        native_matmul(self, &a32, &b32)
    }

    fn matmul_bias(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        bias: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_matmul_bias(self, a, b, bias)
    }
}
