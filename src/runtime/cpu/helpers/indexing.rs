//! Indexing operation helpers for CPU tensors

use super::super::kernels;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Gather elements along a dimension using an index tensor.
pub fn gather_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    index: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // Validate index dtype
    if index.dtype() != DType::I64 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: index.dtype(),
        });
    }

    // Validate index dimensions
    if index.ndim() != ndim {
        return Err(Error::ShapeMismatch {
            expected: shape.to_vec(),
            got: index.shape().to_vec(),
        });
    }

    // Output shape is same as index shape
    let out_shape = index.shape().to_vec();

    let a_contig = ensure_contiguous(a);
    let index_contig = ensure_contiguous(index);
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);

    let a_ptr = a_contig.storage().ptr();
    let index_ptr = index_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::gather_kernel::<T>(
                a_ptr as *const T,
                index_ptr as *const i64,
                out_ptr as *mut T,
                shape,
                &out_shape,
                dim,
            );
        }
    }, "gather");

    Ok(out)
}

/// Scatter values into a tensor at positions specified by an index tensor.
pub fn scatter_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    index: &Tensor<CpuRuntime>,
    src: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // Validate dtypes
    if index.dtype() != DType::I64 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: index.dtype(),
        });
    }

    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    // Validate shapes
    if index.shape() != src.shape() {
        return Err(Error::ShapeMismatch {
            expected: src.shape().to_vec(),
            got: index.shape().to_vec(),
        });
    }

    let a_contig = ensure_contiguous(a);
    let index_contig = ensure_contiguous(index);
    let src_contig = ensure_contiguous(src);
    let out = Tensor::<CpuRuntime>::empty(shape, dtype, &client.device);

    let a_ptr = a_contig.storage().ptr();
    let index_ptr = index_contig.storage().ptr();
    let src_ptr = src_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::scatter_kernel::<T>(
                a_ptr as *const T,
                index_ptr as *const i64,
                src_ptr as *const T,
                out_ptr as *mut T,
                shape,
                index.shape(),
                dim,
            );
        }
    }, "scatter");

    Ok(out)
}

/// Select elements along a dimension using a 1D index tensor.
pub fn index_select_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    index: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // Validate index dtype
    if index.dtype() != DType::I64 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: index.dtype(),
        });
    }

    // Index must be 1D
    if index.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![index.numel()],
            got: index.shape().to_vec(),
        });
    }

    let index_len = index.shape()[0];

    // Output shape: replace dimension `dim` with index length
    let mut out_shape = shape.to_vec();
    out_shape[dim] = index_len;

    let a_contig = ensure_contiguous(a);
    let index_contig = ensure_contiguous(index);

    // Validate all indices are within bounds (before calling unsafe kernel)
    let dim_size = shape[dim];
    let index_data = unsafe {
        std::slice::from_raw_parts(index_contig.storage().ptr() as *const i64, index_len)
    };
    for &idx in index_data.iter() {
        // Negative indices are not supported - must be in [0, dim_size)
        if idx < 0 || idx as usize >= dim_size {
            return Err(Error::IndexOutOfBounds {
                index: if idx < 0 { 0 } else { idx as usize },
                size: dim_size,
            });
        }
    }

    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);

    let a_ptr = a_contig.storage().ptr();
    let index_ptr = index_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::index_select_kernel::<T>(
                a_ptr as *const T,
                index_ptr as *const i64,
                out_ptr as *mut T,
                shape,
                dim,
                index_len,
            );
        }
    }, "index_select");

    Ok(out)
}

/// Put values at specified indices along a dimension.
pub fn index_put_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    index: &Tensor<CpuRuntime>,
    src: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // Validate index dtype
    if index.dtype() != DType::I64 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: index.dtype(),
        });
    }

    // Index must be 1D
    if index.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![index.numel()],
            got: index.shape().to_vec(),
        });
    }

    // Validate src dtype matches
    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    let index_len = index.shape()[0];

    // Validate src shape: must match a's shape except at dim where it equals index_len
    let mut expected_src_shape = shape.to_vec();
    expected_src_shape[dim] = index_len;
    if src.shape() != expected_src_shape {
        return Err(Error::ShapeMismatch {
            expected: expected_src_shape,
            got: src.shape().to_vec(),
        });
    }

    let a_contig = ensure_contiguous(a);
    let index_contig = ensure_contiguous(index);
    let src_contig = ensure_contiguous(src);

    // Validate all indices are within bounds (before calling unsafe kernel)
    let dim_size = shape[dim];
    let index_data = unsafe {
        std::slice::from_raw_parts(index_contig.storage().ptr() as *const i64, index_len)
    };
    for &idx in index_data.iter() {
        // Negative indices are not supported - must be in [0, dim_size)
        if idx < 0 || idx as usize >= dim_size {
            return Err(Error::IndexOutOfBounds {
                index: if idx < 0 { 0 } else { idx as usize },
                size: dim_size,
            });
        }
    }

    // Clone a's data for output
    let out = Tensor::<CpuRuntime>::empty(shape, dtype, &client.device);

    let a_ptr = a_contig.storage().ptr();
    let index_ptr = index_contig.storage().ptr();
    let src_ptr = src_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::index_put_kernel::<T>(
                a_ptr as *const T,
                index_ptr as *const i64,
                src_ptr as *const T,
                out_ptr as *mut T,
                shape,
                dim,
                index_len,
            );
        }
    }, "index_put");

    Ok(out)
}

/// Select elements where mask is true, returning a flattened 1D tensor.
pub fn masked_select_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    mask: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();

    // Validate mask dtype
    if mask.dtype() != DType::U8 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U8,
            rhs: mask.dtype(),
        });
    }

    // Broadcast mask to input shape
    let mask_broadcast = mask.broadcast_to(a.shape())?;

    let a_contig = ensure_contiguous(a);
    let mask_contig = ensure_contiguous(&mask_broadcast);

    let numel = a.numel();
    let a_ptr = a_contig.storage().ptr();
    let mask_ptr = mask_contig.storage().ptr();

    // Count true elements first
    let count = unsafe { kernels::masked_count_kernel(mask_ptr as *const u8, numel) };

    // Allocate output with correct size
    let out = Tensor::<CpuRuntime>::empty(&[count], dtype, &client.device);
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::masked_select_kernel::<T>(
                a_ptr as *const T,
                mask_ptr as *const u8,
                out_ptr as *mut T,
                numel,
            );
        }
    }, "masked_select");

    Ok(out)
}

/// Fill elements where mask is true with a scalar value.
pub fn masked_fill_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    mask: &Tensor<CpuRuntime>,
    value: f64,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();

    // Validate mask dtype
    if mask.dtype() != DType::U8 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U8,
            rhs: mask.dtype(),
        });
    }

    // Broadcast mask to input shape
    let mask_broadcast = mask.broadcast_to(a.shape())?;

    let a_contig = ensure_contiguous(a);
    let mask_contig = ensure_contiguous(&mask_broadcast);
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device);

    let numel = a.numel();
    let a_ptr = a_contig.storage().ptr();
    let mask_ptr = mask_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::masked_fill_kernel::<T>(
                a_ptr as *const T,
                mask_ptr as *const u8,
                out_ptr as *mut T,
                numel,
                value,
            );
        }
    }, "masked_fill");

    Ok(out)
}

/// Look up embeddings from an embedding table using indices.
///
/// # Algorithm
/// For each index i in the flattened indices tensor:
///   output[i, :] = embeddings[indices[i], :]
///
/// Output shape: indices.shape() + [embedding_dim]
pub fn embedding_lookup_impl(
    client: &CpuClient,
    embeddings: &Tensor<CpuRuntime>,
    indices: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = embeddings.dtype();
    let emb_shape = embeddings.shape();

    // Validate embeddings is 2D
    if emb_shape.len() != 2 {
        return Err(Error::ShapeMismatch {
            expected: vec![0, 0], // Indicates 2D expected
            got: emb_shape.to_vec(),
        });
    }

    // Validate indices dtype
    if indices.dtype() != DType::I64 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: indices.dtype(),
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
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);

    let emb_ptr = emb_contig.storage().ptr();
    let idx_ptr = idx_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::embedding_lookup_kernel::<T>(
                emb_ptr as *const T,
                idx_ptr as *const i64,
                out_ptr as *mut T,
                num_indices,
                vocab_size,
                embedding_dim,
            );
        }
    }, "embedding_lookup");

    Ok(out)
}
