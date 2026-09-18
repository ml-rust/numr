//! Walsh-Hadamard transform for WebGPU.
//!
//! Two passes over the flattened buffer. The local pass loads one
//! `chunk`-wide tile per workgroup into workgroup memory, applies `signs` and
//! the `1/sqrt(block_size)` scale, and runs every butterfly stride below the
//! tile width. When `block_size` exceeds the tile, one in-place global
//! dispatch per remaining stride finishes the transform.

use super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::common::validate_fwht_args;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::{FWHT_MAX_CHUNK, FWHT_MIN_CHUNK, fwht};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Invocations per workgroup the local shader is written for.
const LOCAL_WORKGROUP: u32 = 256;

/// Picks the local-pass tile width from the device's workgroup limits.
///
/// Starts at [`FWHT_MAX_CHUNK`] and halves until the f32 tile fits
/// `max_compute_workgroup_storage_size`, never below [`FWHT_MIN_CHUNK`].
/// Then clamps to `block_size` from below by 256 so small blocks still fill
/// a workgroup and large blocks get the widest tile.
fn select_chunk(client: &WgpuClient, block_size: usize) -> Result<usize> {
    let limits = client.wgpu_device().limits();
    if limits.max_compute_invocations_per_workgroup < LOCAL_WORKGROUP
        || limits.max_compute_workgroup_size_x < LOCAL_WORKGROUP
    {
        return Err(Error::backend_limitation(
            "WebGPU",
            "fwht",
            format!(
                "device allows {} invocations per workgroup, kernel needs {LOCAL_WORKGROUP}",
                limits
                    .max_compute_invocations_per_workgroup
                    .min(limits.max_compute_workgroup_size_x)
            ),
        ));
    }

    let storage_bytes = limits.max_compute_workgroup_storage_size as usize;
    let mut cap = FWHT_MAX_CHUNK;
    while cap * std::mem::size_of::<f32>() > storage_bytes && cap > FWHT_MIN_CHUNK {
        cap /= 2;
    }
    if cap * std::mem::size_of::<f32>() > storage_bytes {
        return Err(Error::backend_limitation(
            "WebGPU",
            "fwht",
            format!(
                "device allows {storage_bytes} bytes of workgroup storage, kernel needs {}",
                FWHT_MIN_CHUNK * std::mem::size_of::<f32>()
            ),
        ));
    }

    Ok(block_size.clamp(FWHT_MIN_CHUNK, cap))
}

/// Native Walsh-Hadamard transform on every `block_size` segment of the last
/// axis. F32 only; output has the shape and dtype of `x`.
pub(crate) fn native_fwht(
    client: &WgpuClient,
    x: &Tensor<WgpuRuntime>,
    block_size: usize,
    signs: Option<&Tensor<WgpuRuntime>>,
) -> Result<Tensor<WgpuRuntime>> {
    let last_dim = validate_fwht_args(x, block_size, signs)?;
    let dtype = x.dtype();
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op: "fwht" });
    }
    let shape = x.shape();

    let out = alloc_output(client, shape, dtype)?;

    // A zero-element tensor has nothing to transform, and `get_tensor_buffer`
    // has no buffer to return for a zero-byte allocation.
    if x.numel() == 0 {
        return Ok(out);
    }

    let x_contig = ensure_contiguous(x)?;
    let signs_contig = signs.map(ensure_contiguous).transpose()?;

    let total = x.numel();
    let rows = total / last_dim;
    let rows_u32 = u32::try_from(rows)
        .map_err(|_| Error::backend_limitation("WebGPU", "fwht", "row count exceeds u32"))?;
    let last_dim_u32 = u32::try_from(last_dim)
        .map_err(|_| Error::backend_limitation("WebGPU", "fwht", "last dim exceeds u32"))?;
    let block_size_u32 = u32::try_from(block_size)
        .map_err(|_| Error::backend_limitation("WebGPU", "fwht", "block_size exceeds u32"))?;
    if u32::try_from(total).is_err() {
        return Err(Error::backend_limitation(
            "WebGPU",
            "fwht",
            "element count exceeds u32",
        ));
    }

    let chunk = select_chunk(client, block_size)?;
    let scale = 1.0f32 / (block_size as f32).sqrt();

    let x_buf = get_tensor_buffer(&x_contig)?;
    let out_buf = get_tensor_buffer(&out)?;
    // Without `signs`, a one-element dummy keeps the binding valid; the
    // shader never reads it because `has_signs` is 0.
    let signs_buf = match &signs_contig {
        Some(s) => get_tensor_buffer(s)?,
        None => std::sync::Arc::new(create_storage_buffer(client, &[1.0f32])),
    };

    let mut params = FwhtParams {
        rows: rows_u32,
        last_dim: last_dim_u32,
        block_size: block_size_u32,
        chunk: chunk as u32,
        h: 0,
        has_signs: u32::from(signs_contig.is_some()),
        scale,
    };
    let params_buf = create_params_buffer(client, &params);

    fwht::launch_fwht_local(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &x_buf,
        &signs_buf,
        &out_buf,
        &params_buf,
        total,
        chunk,
        dtype,
    )?;

    // Strides the tile could not hold: chunk, 2*chunk, ..., block_size / 2.
    let pairs = total / 2;
    let mut h = chunk;
    while h < block_size {
        params.h = h as u32;
        let stride_params = create_params_buffer(client, &params);
        fwht::launch_fwht_global(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &out_buf,
            &stride_params,
            pairs,
            dtype,
        )?;
        h *= 2;
    }

    Ok(out)
}
