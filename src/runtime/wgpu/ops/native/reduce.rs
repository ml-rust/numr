//! Reduction operation implementations for WebGPU.

use super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ScalarOps;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::reduce;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

pub(crate) fn native_reduce_op(
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
        // sorted_dims is in descending order, so we remove from highest to lowest
        // to avoid index shifting issues
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

pub(crate) fn native_softmax(
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

pub(crate) fn native_argreduce_op(
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

    // Output indices as I32 (WebGPU doesn't support I64, shader uses u32)
    let out = alloc_output(client, &out_shape, DType::I32);

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
