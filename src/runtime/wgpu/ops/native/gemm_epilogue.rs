//! Native WGPU GEMM epilogue operations.

use super::helpers::*;
use crate::error::{Error, Result};
use crate::ops::{GemmActivation, matmul_bias_output_shape, validate_matmul_bias_dtypes};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::gemm_epilogue;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

pub(crate) fn native_gemm_bias_activation(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    bias: &Tensor<WgpuRuntime>,
    activation: GemmActivation,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = validate_matmul_bias_dtypes(a.dtype(), b.dtype(), bias.dtype())?;
    let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape())
        .ok_or_else(|| Error::shape_mismatch(a.shape(), b.shape()))?;

    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() == 2 && b_shape.len() == 2 {
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let a_c = ensure_contiguous(a)?;
        let b_c = ensure_contiguous(b)?;
        let bias_c = ensure_contiguous(bias)?;
        let out = alloc_output(client, &out_shape, dtype);

        let a_buf = get_tensor_buffer(&a_c)?;
        let b_buf = get_tensor_buffer(&b_c)?;
        let bias_buf = get_tensor_buffer(&bias_c)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params_buf = gemm_epilogue::create_epilogue_params_buffer(
            client.pipeline_cache(),
            m as u32,
            k as u32,
            n as u32,
            1,
            activation,
        );

        gemm_epilogue::launch_gemm_bias_act(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &bias_buf,
            &out_buf,
            &params_buf,
            m,
            n,
            dtype,
        )?;

        return Ok(out);
    }

    if a_shape.len() == 3 && b_shape.len() == 3 {
        let batch_size = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];

        if b_shape[0] != batch_size {
            return Err(Error::ShapeMismatch {
                expected: vec![batch_size, m, k],
                got: b_shape.to_vec(),
            });
        }

        let a_c = ensure_contiguous(a)?;
        let b_c = ensure_contiguous(b)?;
        let bias_c = ensure_contiguous(bias)?;
        let out = alloc_output(client, &out_shape, dtype);

        let a_buf = get_tensor_buffer(&a_c)?;
        let b_buf = get_tensor_buffer(&b_c)?;
        let bias_buf = get_tensor_buffer(&bias_c)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params_buf = gemm_epilogue::create_epilogue_params_buffer(
            client.pipeline_cache(),
            m as u32,
            k as u32,
            n as u32,
            batch_size as u32,
            activation,
        );

        gemm_epilogue::launch_gemm_bias_act_batched(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &bias_buf,
            &out_buf,
            &params_buf,
            m,
            n,
            batch_size,
            dtype,
        )?;

        return Ok(out);
    }

    Err(Error::BackendLimitation {
        backend: "WebGPU",
        operation: "gemm_bias_activation",
        reason: format!(
            "only supports 2D and 3D tensors, got shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        ),
    })
}

pub(crate) fn native_gemm_bias_residual(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    bias: &Tensor<WgpuRuntime>,
    residual: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = validate_matmul_bias_dtypes(a.dtype(), b.dtype(), bias.dtype())?;
    if residual.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: residual.dtype(),
        });
    }

    let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape())
        .ok_or_else(|| Error::shape_mismatch(a.shape(), b.shape()))?;

    if residual.shape() != out_shape.as_slice() {
        return Err(Error::ShapeMismatch {
            expected: out_shape.clone(),
            got: residual.shape().to_vec(),
        });
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape.len() == 2 && b_shape.len() == 2 {
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let a_c = ensure_contiguous(a)?;
        let b_c = ensure_contiguous(b)?;
        let bias_c = ensure_contiguous(bias)?;
        let res_c = ensure_contiguous(residual)?;
        let out = alloc_output(client, &out_shape, dtype);

        let a_buf = get_tensor_buffer(&a_c)?;
        let b_buf = get_tensor_buffer(&b_c)?;
        let bias_buf = get_tensor_buffer(&bias_c)?;
        let res_buf = get_tensor_buffer(&res_c)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params_buf = gemm_epilogue::create_residual_params_buffer(
            client.pipeline_cache(),
            m as u32,
            k as u32,
            n as u32,
            1,
        );

        gemm_epilogue::launch_gemm_bias_residual(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &bias_buf,
            &res_buf,
            &out_buf,
            &params_buf,
            m,
            n,
            dtype,
        )?;

        return Ok(out);
    }

    if a_shape.len() == 3 && b_shape.len() == 3 {
        let batch_size = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];

        if b_shape[0] != batch_size {
            return Err(Error::ShapeMismatch {
                expected: vec![batch_size, m, k],
                got: b_shape.to_vec(),
            });
        }

        let a_c = ensure_contiguous(a)?;
        let b_c = ensure_contiguous(b)?;
        let bias_c = ensure_contiguous(bias)?;
        let res_c = ensure_contiguous(residual)?;
        let out = alloc_output(client, &out_shape, dtype);

        let a_buf = get_tensor_buffer(&a_c)?;
        let b_buf = get_tensor_buffer(&b_c)?;
        let bias_buf = get_tensor_buffer(&bias_c)?;
        let res_buf = get_tensor_buffer(&res_c)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params_buf = gemm_epilogue::create_residual_params_buffer(
            client.pipeline_cache(),
            m as u32,
            k as u32,
            n as u32,
            batch_size as u32,
        );

        gemm_epilogue::launch_gemm_bias_residual_batched(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &bias_buf,
            &res_buf,
            &out_buf,
            &params_buf,
            m,
            n,
            batch_size,
            dtype,
        )?;

        return Ok(out);
    }

    Err(Error::BackendLimitation {
        backend: "WebGPU",
        operation: "gemm_bias_residual",
        reason: format!(
            "only supports 2D and 3D tensors, got shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        ),
    })
}
