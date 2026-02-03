//! Matrix multiplication operation implementations for WebGPU.

use super::helpers::*;
use crate::error::Error;
use crate::error::Result;
use crate::ops::{matmul_bias_output_shape, matmul_output_shape, validate_matmul_bias_dtypes};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::matmul;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

pub(crate) fn native_matmul(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    let out_shape = matmul_output_shape(a.shape(), b.shape())
        .ok_or_else(|| Error::shape_mismatch(a.shape(), b.shape()))?;

    let a_shape = a.shape();
    let b_shape = b.shape();

    // Handle 2D case
    if a_shape.len() == 2 && b_shape.len() == 2 {
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);

        let out = alloc_output(client, &out_shape, dtype);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let b_buf = get_tensor_buffer(&b_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            batch_size: 1,
        };
        let params_buf = create_params_buffer(client, &params);

        // Use tiled for larger matrices, simple for small ones
        if m * n > 256 * 256 {
            matmul::launch_matmul(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &a_buf,
                &b_buf,
                &out_buf,
                &params_buf,
                m,
                n,
                dtype,
            )?;
        } else {
            matmul::launch_matmul_simple(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &a_buf,
                &b_buf,
                &out_buf,
                &params_buf,
                m,
                n,
                dtype,
            )?;
        }

        return Ok(out);
    }

    // Handle batched (3D) matmul natively
    if a_shape.len() == 3 && b_shape.len() == 3 {
        let batch_size = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];

        // Validate batch dimensions match
        if b_shape[0] != batch_size {
            return Err(Error::ShapeMismatch {
                expected: vec![batch_size, m, k],
                got: b_shape.to_vec(),
            });
        }

        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);

        let out = alloc_output(client, &out_shape, dtype);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let b_buf = get_tensor_buffer(&b_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            batch_size: batch_size as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        matmul::launch_batched_matmul(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &out_buf,
            &params_buf,
            m,
            n,
            batch_size,
            dtype,
        )?;

        return Ok(out);
    }

    // >3D tensors are not supported - return error instead of silent fallback
    // (WebGPU shader dispatch is limited to 3D workgroups)
    Err(Error::BackendLimitation {
        backend: "WebGPU",
        operation: "matmul",
        reason: format!(
            "only supports 2D and 3D tensors, got shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        ),
    })
}

/// Native WGPU implementation of fused matrix multiplication with bias.
///
/// Computes C = A @ B + bias where bias is a 1D tensor [N] broadcast across all rows.
/// The bias addition is fused into the GEMM epilogue for efficiency.
pub(crate) fn native_matmul_bias(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    bias: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    // Validate dtypes using unified helper (ensures consistent error handling across backends)
    let dtype = validate_matmul_bias_dtypes(a.dtype(), b.dtype(), bias.dtype())?;

    // Validate shapes and compute output shape
    let out_shape = matmul_bias_output_shape(a.shape(), b.shape(), bias.shape())
        .ok_or_else(|| Error::shape_mismatch(a.shape(), b.shape()))?;

    let a_shape = a.shape();
    let b_shape = b.shape();

    // Handle 2D case
    if a_shape.len() == 2 && b_shape.len() == 2 {
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);
        let bias_contig = ensure_contiguous(bias);

        let out = alloc_output(client, &out_shape, dtype);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let b_buf = get_tensor_buffer(&b_contig)?;
        let bias_buf = get_tensor_buffer(&bias_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            batch_size: 1,
        };
        let params_buf = create_params_buffer(client, &params);

        matmul::launch_matmul_bias(
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

    // Handle batched matmul_bias (3D tensors)
    if a_shape.len() == 3 && b_shape.len() == 3 {
        let batch_size = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];

        // Validate batch dimensions match
        if b_shape[0] != batch_size {
            return Err(Error::ShapeMismatch {
                expected: vec![batch_size, m, k],
                got: b_shape.to_vec(),
            });
        }

        let a_contig = ensure_contiguous(a);
        let b_contig = ensure_contiguous(b);
        let bias_contig = ensure_contiguous(bias);

        let out = alloc_output(client, &out_shape, dtype);

        let a_buf = get_tensor_buffer(&a_contig)?;
        let b_buf = get_tensor_buffer(&b_contig)?;
        let bias_buf = get_tensor_buffer(&bias_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            batch_size: batch_size as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        matmul::launch_batched_matmul_bias(
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

    // >3D tensors are not supported - return error instead of silent fallback
    // (WebGPU shader dispatch is limited to 3D workgroups)
    Err(Error::BackendLimitation {
        backend: "WebGPU",
        operation: "matmul_bias",
        reason: format!(
            "only supports 2D and 3D tensors, got shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        ),
    })
}
