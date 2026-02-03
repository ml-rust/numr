//! Binary and scalar operation implementations for WebGPU.

use super::helpers::*;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::wgpu::shaders::elementwise;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{compute_broadcast_shape, ensure_contiguous, validate_binary_dtypes};
use crate::tensor::Tensor;

pub(crate) fn native_binary_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    // Use shared helpers for validation (same as CPU and CUDA backends)
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    let a_contig = ensure_contiguous(a);
    let b_contig = ensure_contiguous(b);

    let numel: usize = out_shape.iter().product();
    let out = alloc_output(client, &out_shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // Use broadcast kernel if shapes differ, element-wise kernel otherwise
    if a.shape() != b.shape() {
        launch_broadcast_binary(
            client,
            op,
            &a_buf,
            &b_buf,
            &out_buf,
            a.shape(),
            b.shape(),
            &out_shape,
            numel,
            dtype,
        )?;
    } else {
        let params = BinaryParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        elementwise::launch_binary_op(
            client.pipeline_cache(),
            client.wgpu_queue(),
            op,
            &a_buf,
            &b_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;
    }

    Ok(out)
}

/// Launch broadcast binary op with stride buffers.
#[allow(clippy::too_many_arguments)]
fn launch_broadcast_binary(
    client: &WgpuClient,
    op: &'static str,
    a_buf: &wgpu::Buffer,
    b_buf: &wgpu::Buffer,
    out_buf: &wgpu::Buffer,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    numel: usize,
    dtype: DType,
) -> Result<()> {
    let ndim = out_shape.len();

    // Compute broadcast strides (0 for broadcast dimensions)
    let a_strides = compute_broadcast_strides(a_shape, out_shape);
    let b_strides = compute_broadcast_strides(b_shape, out_shape);

    // Compute output strides (row-major)
    let mut out_strides = vec![1u32; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1] as u32;
    }

    // Create stride buffers
    let a_strides_buf = create_storage_buffer(client, &a_strides);
    let b_strides_buf = create_storage_buffer(client, &b_strides);
    let out_strides_buf = create_storage_buffer(client, &out_strides);

    // Create params: [numel, ndim]
    let params = BroadcastBinaryParams {
        numel: numel as u32,
        ndim: ndim as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_broadcast_binary_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        op,
        a_buf,
        b_buf,
        out_buf,
        &a_strides_buf,
        &b_strides_buf,
        &out_strides_buf,
        &params_buf,
        numel,
        dtype,
    )
}

pub(crate) fn native_scalar_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    scalar: f64,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = ScalarParams {
        numel: numel as u32,
        scalar: scalar as f32,
    };
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_scalar_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        op,
        &a_buf,
        &out_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}
