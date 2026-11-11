//! Adjacent binary32 encodings through integer-only WebGPU storage.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::NextRepresentableOps;
use crate::runtime::wgpu::ops::helpers::{alloc_output, get_tensor_buffer};
use crate::runtime::wgpu::shaders::linear_grid::element_count_u32;
use crate::runtime::wgpu::shaders::next_representable::launch_next_representable;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{Device, RuntimeClient};
use crate::tensor::Tensor;

impl NextRepresentableOps<WgpuRuntime> for WgpuClient {
    fn next_up(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        next_representable(self, a, true, "next_up")
    }

    fn next_down(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        next_representable(self, a, false, "next_down")
    }
}

fn next_representable(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    upward: bool,
    op: &'static str,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = input.dtype();
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    if !client.device().is_same(input.device()) {
        return Err(Error::DeviceMismatch);
    }
    element_count_u32(input.numel(), op)?;
    let output = alloc_output(client, input.shape(), dtype)?;
    if input.numel() == 0 {
        return Ok(output);
    }
    let output_buffer = get_tensor_buffer(&output)?;
    launch_next_representable(client, input, &output_buffer, upward, op)?;
    Ok(output)
}
