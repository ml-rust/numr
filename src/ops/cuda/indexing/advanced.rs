//! Advanced indexing operations for CUDA runtime

use crate::algorithm::linalg::helpers::{linalg_demote, linalg_promote};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ScatterReduceOp;
use crate::ops::common::validate_scatter_extents;
use crate::runtime::cuda::kernels::{
    ScatterReduceOpCuda, launch_copy, launch_embedding_lookup, launch_fill_with_f64,
    launch_gather_nd, launch_scatter_reduce,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{compute_contiguous_strides, ensure_contiguous};
use crate::tensor::Tensor;

use super::helpers::normalize_indices_to_i64;

/// Execute embedding_lookup operation.
pub fn embedding_lookup(
    client: &CudaClient,
    embeddings: &Tensor<CudaRuntime>,
    indices: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = embeddings.dtype();
    let emb_shape = embeddings.shape();

    // Validate embeddings is 2D
    if emb_shape.len() != 2 {
        return Err(Error::ShapeMismatch {
            expected: vec![0, 0], // Indicates 2D expected
            got: emb_shape.to_vec(),
        });
    }

    let indices_i64 = normalize_indices_to_i64(client, indices)?;

    let vocab_size = emb_shape[0];
    let embedding_dim = emb_shape[1];
    let num_indices = indices_i64.numel();

    // Output shape: indices.shape() + [embedding_dim]
    let mut out_shape = indices_i64.shape().to_vec();
    out_shape.push(embedding_dim);

    let emb_contig = ensure_contiguous(embeddings)?;
    let idx_contig = ensure_contiguous(&indices_i64)?;
    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

    unsafe {
        launch_embedding_lookup(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            emb_contig.ptr(),
            idx_contig.ptr(),
            out.ptr(),
            num_indices,
            vocab_size,
            embedding_dim,
        )?;
    }

    Ok(out)
}

/// Execute scatter_reduce operation.
pub fn scatter_reduce(
    client: &CudaClient,
    dst: &Tensor<CudaRuntime>,
    dim: usize,
    index: &Tensor<CudaRuntime>,
    src: &Tensor<CudaRuntime>,
    op: ScatterReduceOp,
    include_self: bool,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = dst.dtype();

    // The float reduce kernels exist for F32 and F64. Narrower floats (F16,
    // BF16, FP8) promote to F32, compute, and demote back.
    if dtype.is_float() && !matches!(dtype, DType::F32 | DType::F64) {
        let (dst_promoted, orig_dtype) = linalg_promote(client, dst)?;
        let (src_promoted, _) = linalg_promote(client, src)?;
        let result = scatter_reduce(
            client,
            &dst_promoted,
            dim,
            index,
            &src_promoted,
            op,
            include_self,
        )?;
        return linalg_demote(client, result, orig_dtype);
    }
    let shape = dst.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let index_i64 = normalize_indices_to_i64(client, index)?;

    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    // Validate that index and src have same shape
    if index_i64.shape() != src.shape() {
        return Err(Error::ShapeMismatch {
            expected: src.shape().to_vec(),
            got: index_i64.shape().to_vec(),
        });
    }

    // Validate that index has same number of dimensions as dst
    if index_i64.ndim() != ndim {
        return Err(Error::ShapeMismatch {
            expected: shape.to_vec(),
            got: index_i64.shape().to_vec(),
        });
    }

    // Off the scatter axis, every source coordinate must address the
    // destination.
    validate_scatter_extents(shape, src.shape(), dim)?;

    let cuda_op = match op {
        ScatterReduceOp::Sum => ScatterReduceOpCuda::Sum,
        ScatterReduceOp::Max => ScatterReduceOpCuda::Max,
        ScatterReduceOp::Min => ScatterReduceOpCuda::Min,
        ScatterReduceOp::Prod => ScatterReduceOpCuda::Prod,
        ScatterReduceOp::Mean => ScatterReduceOpCuda::Mean,
    };

    let dst_contig = ensure_contiguous(dst)?;
    let index_contig = ensure_contiguous(&index_i64)?;
    let src_contig = ensure_contiguous(src)?;

    // Allocate output and initialize with dst values if include_self
    let out = Tensor::<CudaRuntime>::empty(shape, dtype, &client.device)?;

    if include_self {
        // Copy dst to output
        unsafe {
            launch_copy(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                dst_contig.ptr(),
                out.ptr(),
                dst.numel(),
            )?;
        }
    } else {
        // Initialize output to identity element for the reduction
        let identity = match op {
            ScatterReduceOp::Sum | ScatterReduceOp::Mean => 0.0,
            ScatterReduceOp::Max => f64::NEG_INFINITY,
            ScatterReduceOp::Min => f64::INFINITY,
            ScatterReduceOp::Prod => 1.0,
        };
        unsafe {
            launch_fill_with_f64(
                &client.context,
                &client.stream,
                client.device.index,
                dtype,
                identity,
                out.ptr(),
                dst.numel(),
            )?;
        }
    }

    unsafe {
        launch_scatter_reduce(
            &client.context,
            &client.stream,
            client.device.index,
            &client.device,
            dtype,
            src_contig.ptr(),
            index_contig.ptr(),
            out.ptr(),
            src.shape(),
            shape,
            dim,
            cuda_op,
            include_self,
        )?;
    }

    Ok(out)
}

/// Execute gather_nd operation.
pub fn gather_nd(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    indices: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = input.dtype();
    let input_shape = input.shape();
    let indices_i64 = normalize_indices_to_i64(client, indices)?;
    let indices_shape = indices_i64.shape();

    // Indices must have at least 1 dimension
    if indices_shape.is_empty() {
        return Err(Error::ShapeMismatch {
            expected: vec![1],
            got: indices_shape.to_vec(),
        });
    }

    // Last dimension of indices is the number of coordinates (M)
    let indices_ndim = indices_shape.len();
    let index_depth = indices_shape[indices_ndim - 1]; // M

    // M must not exceed input dimensions
    if index_depth > input_shape.len() {
        return Err(Error::InvalidDimension {
            dim: index_depth as isize,
            ndim: input_shape.len(),
        });
    }

    // Compute output shape: indices.shape[:-1] + input.shape[M:]
    let mut out_shape: Vec<usize> = indices_shape[..indices_ndim - 1].to_vec();
    out_shape.extend_from_slice(&input_shape[index_depth..]);

    // Handle scalar output case
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    // Compute num_slices (product of indices.shape[:-1]) and slice_size (product of
    // input.shape[M:]). Never clamp either with `.max(1)`: a 1-D `indices` and a
    // fully-consumed `input_shape` already product to 1 over their empty slices, so
    // a clamp only fires on a genuinely zero extent — and then defeats the
    // launcher's own `total == 0` early return, reading index vectors that do not
    // exist or writing to a zero-element output.
    let num_slices: usize = indices_shape[..indices_ndim - 1].iter().product();
    let slice_size: usize = input_shape[index_depth..].iter().product();

    let input_contig = ensure_contiguous(input)?;
    let indices_contig = ensure_contiguous(&indices_i64)?;
    let out = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device)?;

    // Prepare shape and stride arrays as host Vecs (passed as scalar args, no device alloc needed)
    let input_shape_u32: Vec<u32> = input_shape.iter().map(|&s| s as u32).collect();
    let input_strides: Vec<usize> = compute_contiguous_strides(input_shape);
    let input_strides_u32: Vec<u32> = input_strides.iter().map(|&s| s as u32).collect();

    let ndim = input_shape.len();

    unsafe {
        launch_gather_nd(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            input_contig.ptr(),
            indices_contig.ptr(),
            out.ptr(),
            &input_shape_u32,
            &input_strides_u32,
            num_slices,
            slice_size,
            index_depth,
            ndim,
        )?;
    }

    Ok(out)
}
