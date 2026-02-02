//! Parametric activation operation implementation for WebGPU.

use super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::activation_launcher;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Native parametric activation operation (leaky_relu, elu).
///
/// These activations take an extra scalar parameter (negative_slope or alpha).
pub(crate) fn native_parametric_activation(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    param: f64,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // Uses ScalarParams to pass the parameter
    let params = ScalarParams {
        numel: numel as u32,
        scalar: param as f32,
    };
    let params_buf = create_params_buffer(client, &params);

    match op {
        "leaky_relu" => {
            activation_launcher::launch_leaky_relu(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &a_buf,
                &out_buf,
                &params_buf,
                numel,
                dtype,
            )?;
        }
        "elu" => {
            activation_launcher::launch_elu(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &a_buf,
                &out_buf,
                &params_buf,
                numel,
                dtype,
            )?;
        }
        _ => {
            return Err(Error::Internal(format!(
                "Unknown parametric activation: {}",
                op
            )));
        }
    }

    Ok(out)
}
