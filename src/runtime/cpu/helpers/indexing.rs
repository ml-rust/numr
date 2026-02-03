//! Indexing operation helpers for CPU tensors

use super::super::kernels;
#[cfg(target_arch = "x86_64")]
use super::super::kernels::simd::index as simd_index;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ScatterReduceOp;
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

    // Use SIMD for f32/f64 on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        // Count true elements using SIMD
        let count = unsafe { simd_index::masked_count(mask_ptr as *const u8, numel) };

        // Allocate output with correct size
        let out = Tensor::<CpuRuntime>::empty(&[count], dtype, &client.device);
        let out_ptr = out.storage().ptr();

        match dtype {
            DType::F32 => {
                unsafe {
                    simd_index::masked_select_f32(
                        a_ptr as *const f32,
                        mask_ptr as *const u8,
                        out_ptr as *mut f32,
                        numel,
                    );
                }
                return Ok(out);
            }
            DType::F64 => {
                unsafe {
                    simd_index::masked_select_f64(
                        a_ptr as *const f64,
                        mask_ptr as *const u8,
                        out_ptr as *mut f64,
                        numel,
                    );
                }
                return Ok(out);
            }
            _ => {
                // Fall through to scalar for other types
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
                return Ok(out);
            }
        }
    }

    // Scalar fallback for non-x86_64
    #[cfg(not(target_arch = "x86_64"))]
    {
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

    // Use SIMD for f32/f64 on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        match dtype {
            DType::F32 => {
                unsafe {
                    simd_index::masked_fill_f32(
                        a_ptr as *const f32,
                        mask_ptr as *const u8,
                        out_ptr as *mut f32,
                        numel,
                        value as f32,
                    );
                }
                return Ok(out);
            }
            DType::F64 => {
                unsafe {
                    simd_index::masked_fill_f64(
                        a_ptr as *const f64,
                        mask_ptr as *const u8,
                        out_ptr as *mut f64,
                        numel,
                        value,
                    );
                }
                return Ok(out);
            }
            _ => {} // Fall through to scalar for other types
        }
    }

    // Scalar fallback for non-x86_64 or non-f32/f64 types
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

/// Scatter values with reduction into a destination tensor.
#[allow(clippy::too_many_arguments)]
pub fn scatter_reduce_impl(
    client: &CpuClient,
    dst: &Tensor<CpuRuntime>,
    dim: usize,
    index: &Tensor<CpuRuntime>,
    src: &Tensor<CpuRuntime>,
    op: ScatterReduceOp,
    include_self: bool,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = dst.dtype();
    let shape = dst.shape();
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

    // Validate that index and src have same shape
    if index.shape() != src.shape() {
        return Err(Error::ShapeMismatch {
            expected: src.shape().to_vec(),
            got: index.shape().to_vec(),
        });
    }

    // Validate that index has same number of dimensions as dst
    if index.ndim() != ndim {
        return Err(Error::ShapeMismatch {
            expected: shape.to_vec(),
            got: index.shape().to_vec(),
        });
    }

    let dst_contig = ensure_contiguous(dst);
    let index_contig = ensure_contiguous(index);
    let src_contig = ensure_contiguous(src);
    let out = Tensor::<CpuRuntime>::empty(shape, dtype, &client.device);

    // Allocate counts buffer for Mean operation
    let dst_numel: usize = shape.iter().product();
    let counts_buffer: Vec<u32> = vec![0; dst_numel];

    let dst_ptr = dst_contig.storage().ptr();
    let index_ptr = index_contig.storage().ptr();
    let src_ptr = src_contig.storage().ptr();
    let out_ptr = out.storage().ptr();
    let counts_ptr = if op == ScatterReduceOp::Mean {
        counts_buffer.as_ptr() as *mut u32
    } else {
        std::ptr::null_mut()
    };

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::scatter_reduce_kernel::<T>(
                dst_ptr as *const T,
                index_ptr as *const i64,
                src_ptr as *const T,
                out_ptr as *mut T,
                counts_ptr,
                shape,
                index.shape(),
                dim,
                op,
                include_self,
            );
        }
    }, "scatter_reduce");

    Ok(out)
}

/// Gather elements using N-dimensional indices.
pub fn gather_nd_impl(
    client: &CpuClient,
    input: &Tensor<CpuRuntime>,
    indices: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = input.dtype();
    let input_shape = input.shape();
    let indices_shape = indices.shape();

    // Validate indices dtype
    if indices.dtype() != DType::I64 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: indices.dtype(),
        });
    }

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

    // Compute output shape:
    // indices.shape[:-1] + input.shape[M:]
    let mut out_shape: Vec<usize> = indices_shape[..indices_ndim - 1].to_vec();
    out_shape.extend_from_slice(&input_shape[index_depth..]);

    // Handle scalar output case
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let input_contig = ensure_contiguous(input);
    let indices_contig = ensure_contiguous(indices);
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);

    let input_ptr = input_contig.storage().ptr();
    let indices_ptr = indices_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::gather_nd_kernel::<T>(
                input_ptr as *const T,
                indices_ptr as *const i64,
                out_ptr as *mut T,
                input_shape,
                indices_shape,
                &out_shape,
            );
        }
    }, "gather_nd");

    Ok(out)
}

/// Count occurrences of each value in an integer tensor.
pub fn bincount_impl(
    client: &CpuClient,
    input: &Tensor<CpuRuntime>,
    weights: Option<&Tensor<CpuRuntime>>,
    minlength: usize,
) -> Result<Tensor<CpuRuntime>> {
    // Validate input is 1D
    if input.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![input.numel()],
            got: input.shape().to_vec(),
        });
    }

    // Validate input is integer type
    let input_dtype = input.dtype();
    if !matches!(input_dtype, DType::I32 | DType::I64) {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: input_dtype,
        });
    }

    // Validate weights if provided
    let out_dtype = if let Some(w) = weights {
        if w.shape() != input.shape() {
            return Err(Error::ShapeMismatch {
                expected: input.shape().to_vec(),
                got: w.shape().to_vec(),
            });
        }
        w.dtype()
    } else {
        DType::I64 // Count output is I64 when no weights
    };

    let input_contig = ensure_contiguous(input);
    let numel = input.numel();

    // Convert input to i64 if needed
    let input_i64: Vec<i64> = if input_dtype == DType::I64 {
        unsafe {
            std::slice::from_raw_parts(input_contig.storage().ptr() as *const i64, numel).to_vec()
        }
    } else {
        // I32 input
        let i32_slice = unsafe {
            std::slice::from_raw_parts(input_contig.storage().ptr() as *const i32, numel)
        };
        i32_slice.iter().map(|&x| x as i64).collect()
    };

    // Find max value to determine output size
    let max_val = unsafe { kernels::max_i64_kernel(input_i64.as_ptr(), numel) };
    if max_val < 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "bincount requires non-negative values".to_string(),
        });
    }
    let output_len = (max_val as usize + 1).max(minlength);

    let out = Tensor::<CpuRuntime>::empty(&[output_len], out_dtype, &client.device);
    let out_ptr = out.storage().ptr();

    if let Some(w) = weights {
        let w_contig = ensure_contiguous(w);
        let w_ptr = w_contig.storage().ptr();

        dispatch_dtype!(out_dtype, T => {
            let success = unsafe {
                kernels::bincount_kernel::<T>(
                    input_i64.as_ptr(),
                    w_ptr as *const T,
                    out_ptr as *mut T,
                    numel,
                    output_len,
                )
            };
            if !success {
                return Err(Error::InvalidArgument {
                    arg: "input",
                    reason: "bincount requires non-negative values".to_string(),
                });
            }
        }, "bincount");
    } else {
        // No weights - output is I64 counts
        let success = unsafe {
            kernels::bincount_kernel::<i64>(
                input_i64.as_ptr(),
                std::ptr::null(),
                out_ptr as *mut i64,
                numel,
                output_len,
            )
        };
        if !success {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "bincount requires non-negative values".to_string(),
            });
        }
    }

    Ok(out)
}
