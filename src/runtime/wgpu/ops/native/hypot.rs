//! Native WebGPU magnitude with floating-point range handled through integer bits.

use super::helpers::{alloc_output, get_tensor_buffer};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::wgpu::shaders::hypot::launch_hypot;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{Device, RuntimeClient, compute_broadcast_shape, validate_binary_dtypes};
use crate::tensor::Tensor;

pub(crate) fn native_hypot(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = validate_binary_dtypes(a, b)?;
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op: "hypot" });
    }
    let shape = compute_broadcast_shape(a, b)?;
    if !client.device().is_same(a.device()) || !client.device().is_same(b.device()) {
        return Err(Error::DeviceMismatch);
    }
    let output = alloc_output(client, &shape, dtype)?;
    if output.numel() == 0 {
        return Ok(output);
    }
    let a = a.broadcast_to(&shape)?;
    let b = b.broadcast_to(&shape)?;
    let output_buffer = get_tensor_buffer(&output)?;
    launch_hypot(client, &a, &b, &output_buffer)?;
    Ok(output)
}
