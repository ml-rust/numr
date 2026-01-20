//! Native GPU operation implementations for WebGPU.
//!
//! This module contains all the native_* helper functions that implement
//! tensor operations using WGSL compute shaders.

use super::super::shaders::{cumulative, elementwise, index, matmul, norm, reduce};
use super::super::{WgpuClient, WgpuRuntime};
use super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{ScalarOps, broadcast_shape, matmul_bias_output_shape, matmul_output_shape};
use crate::runtime::{RuntimeClient, compute_contiguous_strides, ensure_contiguous};
use crate::tensor::Tensor;
use wgpu::BufferUsages;

pub(super) fn native_binary_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    // Compute broadcast shape
    let out_shape = broadcast_shape(a.shape(), b.shape()).ok_or_else(|| Error::BroadcastError {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })?;

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

pub(super) fn native_unary_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = UnaryParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_unary_op(
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

pub(super) fn native_scalar_op(
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

/// Native parametric activation operation (leaky_relu, elu).
///
/// These activations take an extra scalar parameter (negative_slope or alpha).
pub(super) fn native_parametric_activation(
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
            elementwise::launch_leaky_relu(
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
            elementwise::launch_elu(
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

pub(super) fn native_compare_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    // Broadcasting not yet implemented natively
    if a.shape() != b.shape() {
        return crate::runtime::fallback::compare_op_fallback(
            a,
            b,
            match op {
                "eq" => crate::ops::CompareOp::Eq,
                "ne" => crate::ops::CompareOp::Ne,
                "lt" => crate::ops::CompareOp::Lt,
                "le" => crate::ops::CompareOp::Le,
                "gt" => crate::ops::CompareOp::Gt,
                "ge" => crate::ops::CompareOp::Ge,
                _ => return Err(Error::Internal(format!("Unknown compare op: {}", op))),
            },
            &client.device_id,
            op,
        );
    }

    let a_contig = ensure_contiguous(a);
    let b_contig = ensure_contiguous(b);
    let numel = a.numel();

    // Output is same dtype (F32 for now, TODO: U8 for proper bool)
    let out = alloc_output(client, a.shape(), dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = BinaryParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_compare_op(
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

/// Native cast operation using WGSL compute shader.
///
/// Supports F32 ↔ I32 ↔ U32 conversions on GPU.
pub(super) fn native_cast_op(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dst_dtype: DType,
) -> Result<Tensor<WgpuRuntime>> {
    let src_dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let numel = a.numel();

    // Allocate output with target dtype
    let out = alloc_output(client, a.shape(), dst_dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = CastParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_cast_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &out_buf,
        &params_buf,
        numel,
        src_dtype,
        dst_dtype,
    )?;

    Ok(out)
}

pub(super) fn native_reduce_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    dims: &[usize],
    keepdim: bool,
) -> Result<Tensor<WgpuRuntime>> {
    let _dtype = a.dtype();
    let shape = a.shape();

    if dims.is_empty() {
        // Full reduction
        return native_full_reduce(client, op, a);
    }

    // For multi-dim reduction, reduce one dimension at a time
    if dims.len() > 1 {
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort_by(|a, b| b.cmp(a)); // Sort in descending order

        let mut result = a.clone();
        for &dim in &sorted_dims {
            result = native_single_dim_reduce(client, op, &result, dim, true)?;
        }

        // Remove dims if !keepdim
        if !keepdim {
            let mut out_shape: Vec<usize> = shape.to_vec();
            for &dim in &sorted_dims {
                out_shape.remove(dim);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            result = result.reshape(&out_shape)?;
        }

        return Ok(result);
    }

    // Single dimension reduction
    let dim = dims[0];
    native_single_dim_reduce(client, op, a, dim, keepdim)
}

fn native_single_dim_reduce(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    dim: usize,
    keepdim: bool,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let a_contig = ensure_contiguous(a);

    // Compute parameters
    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();
    let numel_out = outer_size * inner_size;

    // Output shape
    let out_shape: Vec<usize> = if keepdim {
        let mut s = shape.to_vec();
        s[dim] = 1;
        s
    } else {
        let mut s: Vec<usize> = shape[..dim].to_vec();
        s.extend_from_slice(&shape[dim + 1..]);
        if s.is_empty() {
            s.push(1);
        }
        s
    };

    let out = alloc_output(client, &out_shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = ReduceParams {
        reduce_size: reduce_size as u32,
        outer_size: outer_size.max(1) as u32,
        inner_size: inner_size.max(1) as u32,
        numel_out: numel_out.max(1) as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    reduce::launch_reduce_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        op,
        &a_buf,
        &out_buf,
        &params_buf,
        numel_out.max(1),
        dtype,
    )?;

    Ok(out)
}

fn native_full_reduce(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let numel = a.numel();

    // For mean, we need to divide by numel at the end
    let is_mean = op == "mean";
    let reduce_op = if is_mean { "sum" } else { op };

    // Two-pass reduction for large arrays
    let workgroup_size = 256;
    let num_workgroups = (numel + workgroup_size - 1) / workgroup_size;

    if num_workgroups <= 1 {
        // Single pass
        let out = alloc_output(client, &[1], dtype);
        let a_buf = get_tensor_buffer(&a_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = FullReduceParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        reduce::launch_full_reduce_op(
            client.pipeline_cache(),
            client.wgpu_queue(),
            reduce_op,
            &a_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        if is_mean {
            return client.div_scalar(&out, numel as f64);
        }
        return Ok(out);
    }

    // Multi-pass: first reduce to num_workgroups values, then reduce again
    let partial = alloc_output(client, &[num_workgroups], dtype);
    let a_buf = get_tensor_buffer(&a_contig)?;
    let partial_buf = get_tensor_buffer(&partial)?;

    let params = FullReduceParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    reduce::launch_full_reduce_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        reduce_op,
        &a_buf,
        &partial_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    // Second pass
    let out = alloc_output(client, &[1], dtype);
    let out_buf = get_tensor_buffer(&out)?;

    let params2 = FullReduceParams {
        numel: num_workgroups as u32,
    };
    let params_buf2 = create_params_buffer(client, &params2);

    reduce::launch_full_reduce_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        reduce_op,
        &partial_buf,
        &out_buf,
        &params_buf2,
        num_workgroups,
        dtype,
    )?;

    if is_mean {
        return client.div_scalar(&out, numel as f64);
    }
    Ok(out)
}

pub(super) fn native_softmax(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: isize,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Normalize dim
    let dim = if dim < 0 {
        (ndim as isize + dim) as usize
    } else {
        dim as usize
    };

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // Softmax is only efficient on last dimension in our implementation
    // For other dimensions, use CPU fallback
    if dim != ndim - 1 {
        return crate::runtime::fallback::softmax_fallback(
            a,
            dim as isize,
            &client.device_id,
            "softmax",
        );
    }

    let a_contig = ensure_contiguous(a);
    let batch_size: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];

    let out = alloc_output(client, shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = SoftmaxParams {
        batch_size: batch_size.max(1) as u32,
        dim_size: dim_size as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    reduce::launch_softmax_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &out_buf,
        &params_buf,
        batch_size.max(1),
        dtype,
    )?;

    Ok(out)
}

pub(super) fn native_argreduce_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    dim: usize,
    keepdim: bool,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let a_contig = ensure_contiguous(a);

    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();
    let numel_out = outer_size * inner_size;

    // Output shape
    let out_shape: Vec<usize> = if keepdim {
        let mut s = shape.to_vec();
        s[dim] = 1;
        s
    } else {
        let mut s: Vec<usize> = shape[..dim].to_vec();
        s.extend_from_slice(&shape[dim + 1..]);
        if s.is_empty() {
            s.push(1);
        }
        s
    };

    // Output is I64 for indices (but we use F32 storage for now)
    // TODO: proper I64 output
    let out = alloc_output(client, &out_shape, DType::F32);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = ArgReduceParams {
        reduce_size: reduce_size as u32,
        outer_size: outer_size.max(1) as u32,
        inner_size: inner_size.max(1) as u32,
        numel_out: numel_out.max(1) as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    reduce::launch_argreduce_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        op,
        &a_buf,
        &out_buf,
        &params_buf,
        numel_out.max(1),
        dtype,
    )?;

    Ok(out)
}

pub(super) fn native_matmul(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    let out_shape = matmul_output_shape(a.shape(), b.shape()).ok_or_else(|| {
        Error::Internal(format!(
            "matmul shape mismatch: {:?} @ {:?}",
            a.shape(),
            b.shape()
        ))
    })?;

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

    // Batched matmul - fall back to CPU for now
    // TODO: implement batched matmul natively
    crate::runtime::fallback::matmul_fallback(a, b, &out_shape, &client.device_id, "matmul")
}

/// Native WGPU implementation of fused matrix multiplication with bias.
///
/// Computes C = A @ B + bias where bias is a 1D tensor [N] broadcast across all rows.
/// The bias addition is fused into the GEMM epilogue for efficiency.
pub(super) fn native_matmul_bias(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    bias: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    use crate::ops::validate_matmul_bias_dtypes;

    // Validate dtypes using unified helper (ensures consistent error handling across backends)
    let dtype = validate_matmul_bias_dtypes(a.dtype(), b.dtype(), bias.dtype())?;

    // Validate shapes and compute output shape
    let out_shape =
        matmul_bias_output_shape(a.shape(), b.shape(), bias.shape()).ok_or_else(|| {
            Error::Internal(format!(
                "matmul_bias shape mismatch: {:?} @ {:?} + {:?}",
                a.shape(),
                b.shape(),
                bias.shape()
            ))
        })?;

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
            return Err(Error::Internal(format!(
                "matmul_bias batch size mismatch: A has {} batches, B has {}",
                batch_size, b_shape[0]
            )));
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
    Err(Error::Internal(format!(
        "matmul_bias only supports 2D and 3D tensors in WebGPU backend, got shapes {:?} and {:?}",
        a.shape(),
        b.shape()
    )))
}

pub(super) fn native_clamp(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    min_val: f64,
    max_val: f64,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a);
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = ClampParams {
        numel: numel as u32,
        min_val: min_val as f32,
        max_val: max_val as f32,
    };
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_clamp_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &out_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}

pub(super) fn native_where_cond(
    client: &WgpuClient,
    cond: &Tensor<WgpuRuntime>,
    x: &Tensor<WgpuRuntime>,
    y: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = x.dtype();

    // All must have same shape for native implementation
    if cond.shape() != x.shape() || x.shape() != y.shape() {
        return crate::runtime::fallback::where_cond_fallback(
            cond,
            x,
            y,
            &client.device_id,
            "where_cond",
        );
    }

    let cond_contig = ensure_contiguous(cond);
    let x_contig = ensure_contiguous(x);
    let y_contig = ensure_contiguous(y);
    let numel = x.numel();

    let out = alloc_output(client, x.shape(), dtype);

    let cond_buf = get_tensor_buffer(&cond_contig)?;
    let x_buf = get_tensor_buffer(&x_contig)?;
    let y_buf = get_tensor_buffer(&y_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = WhereParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_where_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &cond_buf,
        &x_buf,
        &y_buf,
        &out_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}

pub(super) fn native_rms_norm(
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

pub(super) fn native_layer_norm(
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

pub(super) fn native_index_select(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: usize,
    indices: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // Indices must be I32 on WebGPU (no I64 support)
    if indices.dtype() != DType::I32 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I32,
            rhs: indices.dtype(),
        });
    }

    let a_contig = ensure_contiguous(a);
    let indices_contig = ensure_contiguous(indices);

    // Compute output shape
    let index_len = indices.numel();
    let mut out_shape = shape.to_vec();
    out_shape[dim] = index_len;

    let outer_size: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product();
    let total_output = outer_size * index_len * inner_size;

    let out = alloc_output(client, &out_shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let indices_buf = get_tensor_buffer(&indices_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = IndexSelectParams {
        outer_size: outer_size.max(1) as u32,
        dim_size: dim_size as u32,
        inner_size: inner_size.max(1) as u32,
        index_len: index_len as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    index::launch_index_select(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &indices_buf,
        &out_buf,
        &params_buf,
        total_output.max(1),
        dtype,
    )?;

    Ok(out)
}

pub(super) fn native_gather(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: usize,
    indices: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    if ndim > 4 {
        return Err(Error::Internal(
            "gather: WebGPU implementation supports max 4 dimensions".to_string(),
        ));
    }

    // Indices must be I32 on WebGPU
    if indices.dtype() != DType::I32 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I32,
            rhs: indices.dtype(),
        });
    }

    // Output shape is same as index shape
    let out_shape = indices.shape().to_vec();
    let total_elements = indices.numel();

    let a_contig = ensure_contiguous(a);
    let indices_contig = ensure_contiguous(indices);

    let out = alloc_output(client, &out_shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let indices_buf = get_tensor_buffer(&indices_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // Pack shape and strides into vec4<u32> format
    let input_strides = compute_contiguous_strides(shape);
    let output_strides = compute_contiguous_strides(&out_shape);

    let mut input_shape_arr = [1u32; 4];
    let mut input_strides_arr = [1u32; 4];
    let mut output_shape_arr = [1u32; 4];
    let mut output_strides_arr = [1u32; 4];

    for i in 0..ndim.min(4) {
        input_shape_arr[i] = shape[i] as u32;
        input_strides_arr[i] = input_strides[i] as u32;
    }
    for i in 0..out_shape.len().min(4) {
        output_shape_arr[i] = out_shape[i] as u32;
        output_strides_arr[i] = output_strides[i] as u32;
    }

    let params = GatherParams {
        ndim: ndim as u32,
        dim: dim as u32,
        total_elements: total_elements as u32,
        _padding: 0,
        input_shape: input_shape_arr,
        input_strides: input_strides_arr,
        output_shape: output_shape_arr,
        output_strides: output_strides_arr,
    };
    let params_buf = create_params_buffer(client, &params);

    index::launch_gather(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &indices_buf,
        &out_buf,
        &params_buf,
        total_elements.max(1),
        dtype,
    )?;

    Ok(out)
}

pub(super) fn native_scatter(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: usize,
    indices: &Tensor<WgpuRuntime>,
    src: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    if ndim > 4 {
        return Err(Error::Internal(
            "scatter: WebGPU implementation supports max 4 dimensions".to_string(),
        ));
    }

    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    // Indices must be I32 on WebGPU
    if indices.dtype() != DType::I32 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I32,
            rhs: indices.dtype(),
        });
    }

    let a_contig = ensure_contiguous(a);
    let indices_contig = ensure_contiguous(indices);
    let src_contig = ensure_contiguous(src);

    let src_shape = src.shape();
    let src_total = src.numel();

    // Output is same shape as input
    let out = alloc_output(client, shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let indices_buf = get_tensor_buffer(&indices_contig)?;
    let src_buf = get_tensor_buffer(&src_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // First, copy input to output
    let copy_params = CopyParams {
        numel: a.numel() as u32,
    };
    let copy_params_buf = create_params_buffer(client, &copy_params);

    index::launch_copy(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &out_buf,
        &copy_params_buf,
        a.numel(),
        dtype,
    )?;

    // Then scatter src values into output
    let output_strides = compute_contiguous_strides(shape);
    let src_strides = compute_contiguous_strides(src_shape);

    let mut output_shape_arr = [1u32; 4];
    let mut output_strides_arr = [1u32; 4];
    let mut src_shape_arr = [1u32; 4];
    let mut src_strides_arr = [1u32; 4];

    for i in 0..ndim.min(4) {
        output_shape_arr[i] = shape[i] as u32;
        output_strides_arr[i] = output_strides[i] as u32;
    }
    for i in 0..src_shape.len().min(4) {
        src_shape_arr[i] = src_shape[i] as u32;
        src_strides_arr[i] = src_strides[i] as u32;
    }

    let params = ScatterParams {
        ndim: ndim as u32,
        dim: dim as u32,
        src_total: src_total as u32,
        _padding: 0,
        output_shape: output_shape_arr,
        output_strides: output_strides_arr,
        src_shape: src_shape_arr,
        src_strides: src_strides_arr,
    };
    let params_buf = create_params_buffer(client, &params);

    index::launch_scatter(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &src_buf,
        &indices_buf,
        &out_buf,
        &params_buf,
        src_total.max(1),
        dtype,
    )?;

    Ok(out)
}

pub(super) fn native_masked_fill(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    mask: &Tensor<WgpuRuntime>,
    value: f64,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let numel = a.numel();

    // Mask must be U32 on WebGPU (no U8 support)
    if mask.dtype() != DType::U32 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U32,
            rhs: mask.dtype(),
        });
    }

    if mask.shape() != a.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: mask.shape().to_vec(),
        });
    }

    let a_contig = ensure_contiguous(a);
    let mask_contig = ensure_contiguous(mask);

    let out = alloc_output(client, a.shape(), dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let mask_buf = get_tensor_buffer(&mask_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = MaskedFillParams {
        numel: numel as u32,
        fill_value: value as f32,
    };
    let params_buf = create_params_buffer(client, &params);

    index::launch_masked_fill(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &mask_buf,
        &out_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}

pub(super) fn native_embedding_lookup(
    client: &WgpuClient,
    embeddings: &Tensor<WgpuRuntime>,
    indices: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = embeddings.dtype();
    let emb_shape = embeddings.shape();

    // Validate embeddings is 2D
    if emb_shape.len() != 2 {
        return Err(Error::ShapeMismatch {
            expected: vec![0, 0], // Indicates 2D expected
            got: emb_shape.to_vec(),
        });
    }

    // Validate indices dtype - WebGPU uses I32 for indices
    if indices.dtype() != DType::I32 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I32,
            rhs: indices.dtype(),
        });
    }

    // Only F32, I32, U32 are supported on WebGPU natively
    if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "embedding_lookup",
        });
    }

    let vocab_size = emb_shape[0];
    let embedding_dim = emb_shape[1];
    let num_indices = indices.numel();

    // Output shape: indices.shape() + [embedding_dim]
    let mut out_shape = indices.shape().to_vec();
    out_shape.push(embedding_dim);

    let emb_contig = ensure_contiguous(embeddings);
    let idx_contig = ensure_contiguous(indices);
    let out = alloc_output(client, &out_shape, dtype);

    let emb_buf = get_tensor_buffer(&emb_contig)?;
    let idx_buf = get_tensor_buffer(&idx_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = EmbeddingLookupParams {
        num_indices: num_indices as u32,
        vocab_size: vocab_size as u32,
        embedding_dim: embedding_dim as u32,
        _pad0: 0,
    };
    let params_buf = create_params_buffer(client, &params);

    index::launch_embedding_lookup(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &emb_buf,
        &idx_buf,
        &out_buf,
        &params_buf,
        num_indices,
        dtype,
    )?;

    Ok(out)
}

pub(super) fn native_masked_select(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    mask: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let numel = a.numel();

    // Mask must be U32 on WebGPU
    if mask.dtype() != DType::U32 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U32,
            rhs: mask.dtype(),
        });
    }

    if mask.shape() != a.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: mask.shape().to_vec(),
        });
    }

    let a_contig = ensure_contiguous(a);
    let mask_contig = ensure_contiguous(mask);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let mask_buf = get_tensor_buffer(&mask_contig)?;

    // Phase 1: Count the number of selected elements
    // Need an atomic buffer for count result
    let count_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("masked_count_result"),
        size: 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Initialize count to 0
    client.queue.write_buffer(&count_buffer, 0, &[0u8; 4]);

    let count_params = MaskedCountParams {
        numel: numel as u32,
    };
    let count_params_buf = create_params_buffer(client, &count_params);

    index::launch_masked_count(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &mask_buf,
        &count_buffer,
        &count_params_buf,
        numel,
        dtype,
    )?;

    // Read count back to CPU (need to synchronize)
    let staging_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("count_staging"),
        size: 4,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_count"),
        });
    encoder.copy_buffer_to_buffer(&count_buffer, 0, &staging_buffer, 0, 4);
    client.queue.submit(std::iter::once(encoder.finish()));

    // Wait for GPU and read the count
    let slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    let _ = client.wgpu_device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(60)),
    });
    receiver.recv().unwrap().unwrap();

    let count = {
        let data = slice.get_mapped_range();
        u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize
    };
    drop(staging_buffer);

    if count == 0 {
        // Return empty tensor
        return Ok(Tensor::empty(&[0], dtype, client.device()));
    }

    // Phase 2: Compute prefix sum
    let prefix_sum_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prefix_sum"),
        size: (numel * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let prefix_params = MaskedCountParams {
        numel: numel as u32,
    };
    let prefix_params_buf = create_params_buffer(client, &prefix_params);

    index::launch_masked_prefix_sum(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &mask_buf,
        &prefix_sum_buffer,
        &prefix_params_buf,
        numel,
        dtype,
    )?;

    // Phase 3: Gather selected elements
    let out = alloc_output(client, &[count], dtype);
    let out_buf = get_tensor_buffer(&out)?;

    let select_params = MaskedSelectParams {
        numel: numel as u32,
    };
    let select_params_buf = create_params_buffer(client, &select_params);

    index::launch_masked_select(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &mask_buf,
        &prefix_sum_buffer,
        &out_buf,
        &select_params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}

// ============================================================================
// Cumulative Operations
// ============================================================================

/// Native cumulative sum along a dimension.
///
/// Computes the cumulative sum of elements along the specified dimension.
/// Output has the same shape as input.
pub(super) fn native_cumsum(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: isize,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if ndim == 0 {
        return Err(Error::InvalidDimension { dim, ndim });
    }

    // Normalize dimension
    let dim = if dim < 0 {
        (ndim as isize + dim) as usize
    } else {
        dim as usize
    };

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let a_contig = ensure_contiguous(a);

    // Compute parameters
    let scan_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    // Output has same shape as input
    let out = alloc_output(client, shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // Use strided kernel if not operating on last dimension
    if dim == ndim - 1 || inner_size == 1 {
        // Contiguous case: operating on last dim or only dim
        let params = CumsumParams {
            scan_size: scan_size as u32,
            outer_size: outer_size.max(1) as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        cumulative::launch_cumsum(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &out_buf,
            &params_buf,
            outer_size.max(1),
            dtype,
        )?;
    } else {
        // Strided case: need inner_size
        let params = CumsumStridedParams {
            scan_size: scan_size as u32,
            outer_size: outer_size.max(1) as u32,
            inner_size: inner_size as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        let total_inner = outer_size.max(1) * inner_size;
        cumulative::launch_cumsum_strided(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &out_buf,
            &params_buf,
            total_inner,
            dtype,
        )?;
    }

    Ok(out)
}

/// Native cumulative product along a dimension.
///
/// Computes the cumulative product of elements along the specified dimension.
/// Output has the same shape as input.
pub(super) fn native_cumprod(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: isize,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if ndim == 0 {
        return Err(Error::InvalidDimension { dim, ndim });
    }

    // Normalize dimension
    let dim = if dim < 0 {
        (ndim as isize + dim) as usize
    } else {
        dim as usize
    };

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let a_contig = ensure_contiguous(a);

    // Compute parameters
    let scan_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    // Output has same shape as input
    let out = alloc_output(client, shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // Use strided kernel if not operating on last dimension
    if dim == ndim - 1 || inner_size == 1 {
        // Contiguous case: operating on last dim or only dim
        let params = CumprodParams {
            scan_size: scan_size as u32,
            outer_size: outer_size.max(1) as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        cumulative::launch_cumprod(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &out_buf,
            &params_buf,
            outer_size.max(1),
            dtype,
        )?;
    } else {
        // Strided case: need inner_size
        let params = CumprodStridedParams {
            scan_size: scan_size as u32,
            outer_size: outer_size.max(1) as u32,
            inner_size: inner_size as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        let total_inner = outer_size.max(1) * inner_size;
        cumulative::launch_cumprod_strided(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &out_buf,
            &params_buf,
            total_inner,
            dtype,
        )?;
    }

    Ok(out)
}

/// Native log-sum-exp reduction.
///
/// Computes log(sum(exp(x))) along specified dimensions in a numerically stable way.
/// Uses the identity: logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
///
/// Only supports floating-point dtypes (F32 on WebGPU, F16 with extension).
pub(super) fn native_logsumexp(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dims: &[usize],
    keepdim: bool,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Logsumexp only makes sense for floating-point types
    if !dtype.is_float() {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "logsumexp",
        });
    }

    // Empty dims means no reduction - return a copy (matches CPU behavior)
    if dims.is_empty() {
        return Ok(a.clone());
    }

    // For multi-dim reduction, reduce one dimension at a time (from highest to lowest)
    if dims.len() > 1 {
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort_by(|a, b| b.cmp(a)); // Descending order

        let mut result = a.clone();
        for &dim in &sorted_dims {
            result = native_logsumexp_single_dim(client, &result, dim, true)?;
        }

        // Remove dims if !keepdim
        if !keepdim {
            let mut out_shape: Vec<usize> = shape.to_vec();
            for &dim in &sorted_dims {
                out_shape.remove(dim);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            result = result.reshape(&out_shape)?;
        }

        return Ok(result);
    }

    // Single dimension reduction
    let dim = dims[0];
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    native_logsumexp_single_dim(client, a, dim, keepdim)
}

/// Logsumexp along a single dimension.
fn native_logsumexp_single_dim(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: usize,
    keepdim: bool,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let a_contig = ensure_contiguous(a);

    // Compute parameters
    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    // Output shape
    let out_shape: Vec<usize> = if keepdim {
        let mut s = shape.to_vec();
        s[dim] = 1;
        s
    } else {
        let mut s: Vec<usize> = shape[..dim].to_vec();
        s.extend_from_slice(&shape[dim + 1..]);
        if s.is_empty() {
            s.push(1);
        }
        s
    };

    let out = alloc_output(client, &out_shape, dtype);

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // Use strided kernel if not operating on last dimension
    if dim == ndim - 1 || inner_size == 1 {
        // Contiguous case
        let params = LogsumexpParams {
            reduce_size: reduce_size as u32,
            outer_size: outer_size.max(1) as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        cumulative::launch_logsumexp(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &out_buf,
            &params_buf,
            outer_size.max(1),
            dtype,
        )?;
    } else {
        // Strided case
        let params = LogsumexpStridedParams {
            reduce_size: reduce_size as u32,
            outer_size: outer_size.max(1) as u32,
            inner_size: inner_size as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        let total_inner = outer_size.max(1) * inner_size;
        cumulative::launch_logsumexp_strided(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &out_buf,
            &params_buf,
            total_inner,
            dtype,
        )?;
    }

    Ok(out)
}
