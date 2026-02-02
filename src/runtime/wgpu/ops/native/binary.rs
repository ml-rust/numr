//! Binary and scalar operation implementations for WebGPU.

use super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::wgpu::shaders::elementwise;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{
    RuntimeClient, compute_broadcast_shape, ensure_contiguous, validate_binary_dtypes,
};
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

    // Broadcasting not yet implemented natively - fall back for different shapes
    if a.shape() != b.shape() {
        return crate::runtime::fallback::binary_op_fallback(
            a,
            b,
            match op {
                "add" => crate::ops::BinaryOp::Add,
                "sub" => crate::ops::BinaryOp::Sub,
                "mul" => crate::ops::BinaryOp::Mul,
                "div" => crate::ops::BinaryOp::Div,
                "pow" => crate::ops::BinaryOp::Pow,
                "maximum" | "max" => crate::ops::BinaryOp::Max,
                "minimum" | "min" => crate::ops::BinaryOp::Min,
                _ => return Err(Error::Internal(format!("Unknown binary op: {}", op))),
            },
            &client.device_id,
            op,
        );
    }

    let a_contig = ensure_contiguous(a);
    let b_contig = ensure_contiguous(b);

    let numel = out_shape.iter().product();
    let out = alloc_output(client, &out_shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

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

    Ok(out)
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
