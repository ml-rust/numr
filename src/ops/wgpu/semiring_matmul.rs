//! Semiring matrix multiplication for WebGPU runtime

use crate::error::Result;
use crate::ops::SemiringMatmulOps;
use crate::ops::semiring::SemiringOp;
use crate::runtime::wgpu::ops::native::native_semiring_matmul;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

impl SemiringMatmulOps<WgpuRuntime> for WgpuClient {
    fn semiring_matmul(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        op: SemiringOp,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_semiring_matmul(self, a, b, op)
    }
}
