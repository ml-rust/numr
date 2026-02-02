//! Normalization operation implementations for WebGPU.

use super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::norm;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

pub(crate) fn native_rms_norm(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    weight: &Tensor<WgpuRuntime>,
    eps: f32,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();

    if shape.len() < 1 {
        return Err(Error::Internal(
            "rms_norm requires at least 1D input".to_string(),
        ));
    }

    let hidden_size = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let a_contig = ensure_contiguous(a);
    let weight_contig = ensure_contiguous(weight);

    let out = alloc_output(client, shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let weight_buf = get_tensor_buffer(&weight_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = RmsNormParams {
        batch_size: batch_size.max(1) as u32,
        hidden_size: hidden_size as u32,
        eps,
    };
    let params_buf = create_params_buffer(client, &params);

    norm::launch_rms_norm(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &weight_buf,
        &out_buf,
        &params_buf,
        batch_size.max(1),
        dtype,
    )?;

    Ok(out)
}

pub(crate) fn native_layer_norm(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    weight: &Tensor<WgpuRuntime>,
    bias: &Tensor<WgpuRuntime>,
    eps: f32,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();

    if shape.len() < 1 {
        return Err(Error::Internal(
            "layer_norm requires at least 1D input".to_string(),
        ));
    }

    let hidden_size = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let a_contig = ensure_contiguous(a);
    let weight_contig = ensure_contiguous(weight);
    let bias_contig = ensure_contiguous(bias);

    let out = alloc_output(client, shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let weight_buf = get_tensor_buffer(&weight_contig)?;
    let bias_buf = get_tensor_buffer(&bias_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = LayerNormParams {
        batch_size: batch_size.max(1) as u32,
        hidden_size: hidden_size as u32,
        eps,
    };
    let params_buf = create_params_buffer(client, &params);

    norm::launch_layer_norm(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &weight_buf,
        &bias_buf,
        &out_buf,
        &params_buf,
        batch_size.max(1),
        dtype,
    )?;

    Ok(out)
}
