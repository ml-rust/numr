//! Snake activation for WebGPU. F32 only.
//!
//! Forward is one dispatch over the `[outer, C, inner]` view; backward is one
//! elementwise dispatch for `d_x` plus one workgroup-per-channel dispatch for
//! `d_alpha` and `d_beta`.

use super::helpers::*;
use crate::error::{Error, Result};
use crate::ops::activation_common::{SnakeGeometry, validate_snake_beta};
use crate::runtime::RuntimeClient;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::snake::{
    SnakeParams, launch_snake_beta, launch_snake_beta_dparams, launch_snake_beta_dx,
};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// The shaders index with `u32`; reject an extent they cannot address.
fn extent_u32(name: &'static str, v: usize) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::InvalidArgument {
        arg: name,
        reason: format!("{name} = {v} exceeds the u32 extent the snake shaders index with"),
    })
}

fn params(extent: usize, geom: SnakeGeometry, eps: f64) -> Result<SnakeParams> {
    Ok(SnakeParams {
        extent: extent_u32("extent", extent)?,
        channels: extent_u32("channels", geom.channels)?,
        inner: extent_u32("inner", geom.inner)?,
        eps: eps as f32,
    })
}

/// `y = x + sin(alpha * x)^2 / (beta + eps)`, channel axis `dim`.
pub(crate) fn native_snake_beta(
    client: &WgpuClient,
    x: &Tensor<WgpuRuntime>,
    alpha: &Tensor<WgpuRuntime>,
    beta: &Tensor<WgpuRuntime>,
    dim: isize,
    eps: f64,
) -> Result<Tensor<WgpuRuntime>> {
    let geom = validate_snake_beta(x, alpha, beta, dim, eps)?;
    let dtype = x.dtype();
    let x_contig = ensure_contiguous(x)?;
    let alpha_contig = ensure_contiguous(alpha)?;
    let beta_contig = ensure_contiguous(beta)?;
    let numel = x.numel();
    let out = alloc_output(client, x.shape(), dtype)?;
    if numel == 0 {
        return Ok(out);
    }

    let x_buf = get_tensor_buffer(&x_contig)?;
    let alpha_buf = get_tensor_buffer(&alpha_contig)?;
    let beta_buf = get_tensor_buffer(&beta_contig)?;
    let out_buf = get_tensor_buffer(&out)?;
    let params_buf = create_params_buffer(client, &params(numel, geom, eps)?);
    launch_snake_beta(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &x_buf,
        &alpha_buf,
        &beta_buf,
        &out_buf,
        &params_buf,
        numel,
        dtype,
    )?;
    Ok(out)
}

/// `(d_x, d_alpha, d_beta)` for [`native_snake_beta`].
#[allow(clippy::type_complexity)]
pub(crate) fn native_snake_beta_bwd(
    client: &WgpuClient,
    grad: &Tensor<WgpuRuntime>,
    x: &Tensor<WgpuRuntime>,
    alpha: &Tensor<WgpuRuntime>,
    beta: &Tensor<WgpuRuntime>,
    dim: isize,
    eps: f64,
) -> Result<(
    Tensor<WgpuRuntime>,
    Tensor<WgpuRuntime>,
    Tensor<WgpuRuntime>,
)> {
    let geom = validate_snake_beta(x, alpha, beta, dim, eps)?;
    let dtype = x.dtype();
    if grad.shape() != x.shape() {
        return Err(Error::ShapeMismatch {
            expected: x.shape().to_vec(),
            got: grad.shape().to_vec(),
        });
    }
    if grad.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: grad.dtype(),
        });
    }
    let grad_contig = ensure_contiguous(grad)?;
    let x_contig = ensure_contiguous(x)?;
    let alpha_contig = ensure_contiguous(alpha)?;
    let beta_contig = ensure_contiguous(beta)?;
    let numel = x.numel();
    let d_x = alloc_output(client, x.shape(), dtype)?;
    if numel == 0 {
        let d_alpha = Tensor::<WgpuRuntime>::zeros(alpha.shape(), dtype, client.device())?;
        let d_beta = Tensor::<WgpuRuntime>::zeros(beta.shape(), dtype, client.device())?;
        return Ok((d_x, d_alpha, d_beta));
    }
    let d_alpha = alloc_output(client, alpha.shape(), dtype)?;
    let d_beta = alloc_output(client, beta.shape(), dtype)?;

    let grad_buf = get_tensor_buffer(&grad_contig)?;
    let x_buf = get_tensor_buffer(&x_contig)?;
    let alpha_buf = get_tensor_buffer(&alpha_contig)?;
    let beta_buf = get_tensor_buffer(&beta_contig)?;
    let d_x_buf = get_tensor_buffer(&d_x)?;
    let d_alpha_buf = get_tensor_buffer(&d_alpha)?;
    let d_beta_buf = get_tensor_buffer(&d_beta)?;

    let dx_params = create_params_buffer(client, &params(numel, geom, eps)?);
    launch_snake_beta_dx(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &grad_buf,
        &x_buf,
        &alpha_buf,
        &beta_buf,
        &d_x_buf,
        &dx_params,
        numel,
        dtype,
    )?;

    let dp_params = create_params_buffer(client, &params(geom.outer, geom, eps)?);
    launch_snake_beta_dparams(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &grad_buf,
        &x_buf,
        &alpha_buf,
        &beta_buf,
        &d_alpha_buf,
        &d_beta_buf,
        &dp_params,
        geom.channels,
        dtype,
    )?;
    Ok((d_x, d_alpha, d_beta))
}
