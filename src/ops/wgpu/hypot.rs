//! Wgpu scaled Euclidean magnitude.

use crate::error::Result;
use crate::ops::HypotOps;
use crate::runtime::wgpu::ops::native::native_hypot;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

impl HypotOps<WgpuRuntime> for WgpuClient {
    fn hypot(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_hypot(self, a, b)
    }
}
