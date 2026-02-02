//! Conditional operations for WebGPU runtime

use crate::error::Result;
use crate::ops::ConditionalOps;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::native::native_where_cond;
use crate::tensor::Tensor;

impl ConditionalOps<WgpuRuntime> for WgpuClient {
    fn where_cond(
        &self,
        cond: &Tensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_where_cond(self, cond, x, y)
    }
}
