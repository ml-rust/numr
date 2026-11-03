//! WebGPU replicate-mode pad: one native compute pass, no host round trip.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::common::pad_replicate::validate_replicate_pad;
use crate::runtime::wgpu::ops::helpers::{
    MAX_DIMS, PadReplicateParams, alloc_output, create_params_buffer, get_tensor_buffer,
    pack_u32_array,
};
use crate::runtime::wgpu::shaders::linear_grid::element_count_u32;
use crate::runtime::wgpu::shaders::shape;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Pad `tensor` by repeating its edge elements. Supports F32, I32 and U32 and
/// at most `MAX_DIMS` dimensions, the same limits as reflect-mode pad.
pub(super) fn pad_replicate(
    client: &WgpuClient,
    tensor: &Tensor<WgpuRuntime>,
    padding: &[usize],
) -> Result<Tensor<WgpuRuntime>> {
    let params = validate_replicate_pad(tensor, padding)?;

    // No-op if all padding is zero
    if params.pad_per_dim.iter().all(|&(b, a)| b == 0 && a == 0) {
        return tensor.contiguous();
    }

    let dtype = tensor.dtype();
    if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "pad_replicate",
        });
    }

    let ndim = params.out_shape.len();
    if ndim > MAX_DIMS {
        return Err(Error::backend_limitation(
            "WebGPU",
            "pad_replicate",
            format!("max {MAX_DIMS} dimensions, got {ndim}"),
        ));
    }

    let tensor_contig = tensor.contiguous()?;

    let total_elements: usize = params.out_shape.iter().product();
    // The output is no smaller than the source, so its count bounds every cast.
    element_count_u32(total_elements, "pad_replicate")?;

    let out = alloc_output(client, &params.out_shape, dtype)?;
    let out_buf = get_tensor_buffer(&out)?;
    let src_buf = get_tensor_buffer(&tensor_contig)?;

    let mut src_shape_flat = [0u32; 8];
    let mut out_shape_flat = [0u32; 8];
    let mut pad_before_flat = [0u32; 8];
    for i in 0..ndim {
        src_shape_flat[i] = tensor.shape()[i] as u32;
        out_shape_flat[i] = params.out_shape[i] as u32;
        pad_before_flat[i] = params.pad_per_dim[i].0 as u32;
    }

    let shader_params = PadReplicateParams {
        ndim: ndim as u32,
        total_elements: total_elements as u32,
        _pad0: 0,
        _pad1: 0,
        src_shape: pack_u32_array(&src_shape_flat),
        out_shape: pack_u32_array(&out_shape_flat),
        pad_before: pack_u32_array(&pad_before_flat),
    };
    let params_buf = create_params_buffer(client, &shader_params);

    shape::launch_pad_replicate(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &src_buf,
        &out_buf,
        &params_buf,
        total_elements,
        dtype,
    )?;

    Ok(out)
}
