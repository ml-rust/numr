//! Index operation kernels (gather, scatter, masked operations)

use crate::dtype::{DType, Element};

/// Gather elements along a dimension using an index tensor.
///
/// For a 3D tensor with dim=1:
/// `out[i][j][k] = input[i][index[i][j][k]][k]`
///
/// # Arguments
/// * `a` - Input data pointer
/// * `indices` - Index tensor pointer (i64 values)
/// * `out` - Output pointer
/// * `shape` - Shape of input tensor
/// * `index_shape` - Shape of index tensor (same as output shape)
/// * `dim` - Dimension along which to gather
///
/// # Safety
/// - All pointers must be valid for the specified shapes
/// - `indices` must contain valid indices within bounds of `shape[dim]`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gather_kernel<T: Element>(
    a: *const T,
    indices: *const i64,
    out: *mut T,
    shape: &[usize],
    index_shape: &[usize],
    dim: usize,
) {
    let ndim = shape.len();
    if ndim == 0 {
        return;
    }

    // Compute strides for input tensor (row-major)
    let mut a_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        a_strides[i] = a_strides[i + 1] * shape[i + 1];
    }

    // Compute strides for index/output tensor (row-major)
    let mut idx_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        idx_strides[i] = idx_strides[i + 1] * index_shape[i + 1];
    }

    let total = index_shape.iter().product::<usize>();

    // Iterate over all output positions
    for out_idx in 0..total {
        // Convert linear index to multi-dimensional indices
        let mut remaining = out_idx;
        let mut multi_idx = vec![0usize; ndim];
        for d in 0..ndim {
            multi_idx[d] = remaining / idx_strides[d];
            remaining %= idx_strides[d];
        }

        // Get the index value from the indices tensor
        let index_val = *indices.add(out_idx);
        if index_val < 0 || index_val as usize >= shape[dim] {
            // Out of bounds - set to zero (could also panic)
            *out.add(out_idx) = T::zero();
            continue;
        }

        // Compute source position: replace multi_idx[dim] with index_val
        let mut src_offset = 0;
        for d in 0..ndim {
            let coord = if d == dim {
                index_val as usize
            } else {
                multi_idx[d]
            };
            src_offset += coord * a_strides[d];
        }

        *out.add(out_idx) = *a.add(src_offset);
    }
}

/// Scatter values into a tensor at positions specified by an index tensor.
///
/// For a 3D tensor with dim=1:
/// `out[i][index[i][j][k]][k] = src[i][j][k]`
///
/// First copies `a` to `out`, then scatters `src` values.
///
/// # Arguments
/// * `a` - Base tensor to scatter into
/// * `indices` - Index tensor pointer (i64 values)
/// * `src` - Source values to scatter
/// * `out` - Output pointer (must be separate from a)
/// * `shape` - Shape of input/output tensor
/// * `index_shape` - Shape of index/src tensors
/// * `dim` - Dimension along which to scatter
///
/// # Safety
/// - All pointers must be valid for the specified shapes
/// - `out` must not alias with `a`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn scatter_kernel<T: Element>(
    a: *const T,
    indices: *const i64,
    src: *const T,
    out: *mut T,
    shape: &[usize],
    index_shape: &[usize],
    dim: usize,
) {
    let ndim = shape.len();
    if ndim == 0 {
        return;
    }

    let a_numel: usize = shape.iter().product();

    // First, copy a to out
    std::ptr::copy_nonoverlapping(a, out, a_numel);

    // Compute strides for output tensor (row-major)
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        out_strides[i] = out_strides[i + 1] * shape[i + 1];
    }

    // Compute strides for index/src tensor (row-major)
    let mut idx_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        idx_strides[i] = idx_strides[i + 1] * index_shape[i + 1];
    }

    let total = index_shape.iter().product::<usize>();

    // Scatter src values to out at index positions
    for src_idx in 0..total {
        // Convert linear index to multi-dimensional indices
        let mut remaining = src_idx;
        let mut multi_idx = vec![0usize; ndim];
        for d in 0..ndim {
            multi_idx[d] = remaining / idx_strides[d];
            remaining %= idx_strides[d];
        }

        // Get the index value from the indices tensor
        let index_val = *indices.add(src_idx);
        if index_val < 0 || index_val as usize >= shape[dim] {
            // Out of bounds - skip
            continue;
        }

        // Compute destination position: replace multi_idx[dim] with index_val
        let mut dst_offset = 0;
        for d in 0..ndim {
            let coord = if d == dim {
                index_val as usize
            } else {
                multi_idx[d]
            };
            dst_offset += coord * out_strides[d];
        }

        *out.add(dst_offset) = *src.add(src_idx);
    }
}

/// Select elements along a dimension using a 1D index tensor.
///
/// Simpler than gather - the index tensor is 1D and applies uniformly
/// to all positions in the specified dimension.
///
/// # Arguments
/// * `a` - Input data pointer
/// * `indices` - 1D index tensor pointer (i64 values), length = index_len
/// * `out` - Output pointer
/// * `shape` - Shape of input tensor
/// * `dim` - Dimension along which to select
/// * `index_len` - Length of the 1D index tensor
///
/// # Safety
/// - All pointers must be valid for the specified shapes
/// - `indices` must contain valid indices within bounds of `shape[dim]`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn index_select_kernel<T: Element>(
    a: *const T,
    indices: *const i64,
    out: *mut T,
    shape: &[usize],
    dim: usize,
    index_len: usize,
) {
    let ndim = shape.len();
    if ndim == 0 {
        return;
    }

    // Compute sizes: outer * dim_size * inner
    let outer_size: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product();

    // For each outer position
    for outer in 0..outer_size.max(1) {
        // For each selected index
        for (sel_idx, &idx_ptr) in std::slice::from_raw_parts(indices, index_len)
            .iter()
            .enumerate()
        {
            let idx = idx_ptr as usize;
            if idx >= dim_size {
                // Out of bounds - fill with zeros
                for inner in 0..inner_size.max(1) {
                    let out_offset =
                        outer * index_len * inner_size.max(1) + sel_idx * inner_size.max(1) + inner;
                    *out.add(out_offset) = T::zero();
                }
                continue;
            }

            // Copy the entire inner slice
            for inner in 0..inner_size.max(1) {
                let src_offset =
                    outer * dim_size * inner_size.max(1) + idx * inner_size.max(1) + inner;
                let out_offset =
                    outer * index_len * inner_size.max(1) + sel_idx * inner_size.max(1) + inner;
                *out.add(out_offset) = *a.add(src_offset);
            }
        }
    }
}

/// Put values at specified indices along a dimension.
///
/// Copies input `a` to output, then overwrites positions specified by `indices`
/// with values from `src`.
///
/// # Arguments
/// * `a` - Input tensor data pointer
/// * `indices` - 1D index tensor pointer (i64)
/// * `src` - Source values to put at indexed positions
/// * `out` - Output data pointer (must be same size as input)
/// * `shape` - Shape of input tensor `a`
/// * `dim` - Dimension along which to put values
/// * `index_len` - Length of the 1D index tensor
///
/// # Safety
/// - All pointers must be valid for the specified shapes
/// - `indices` must contain valid indices within bounds of `shape[dim]`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn index_put_kernel<T: Element>(
    a: *const T,
    indices: *const i64,
    src: *const T,
    out: *mut T,
    shape: &[usize],
    dim: usize,
    index_len: usize,
) {
    let ndim = shape.len();
    if ndim == 0 {
        return;
    }

    // Compute sizes: outer * dim_size * inner
    let outer_size: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product();

    // First, copy all of a to out
    let total_size: usize = shape.iter().product();
    std::ptr::copy_nonoverlapping(a, out, total_size);

    // Now overwrite the indexed positions with src values
    for outer in 0..outer_size.max(1) {
        for (sel_idx, &idx_ptr) in std::slice::from_raw_parts(indices, index_len)
            .iter()
            .enumerate()
        {
            let idx = idx_ptr as usize;
            if idx >= dim_size {
                // Out of bounds - skip
                continue;
            }

            // Overwrite the entire inner slice at this index
            for inner in 0..inner_size.max(1) {
                let out_offset =
                    outer * dim_size * inner_size.max(1) + idx * inner_size.max(1) + inner;
                let src_offset =
                    outer * index_len * inner_size.max(1) + sel_idx * inner_size.max(1) + inner;
                *out.add(out_offset) = *src.add(src_offset);
            }
        }
    }
}

/// Count elements where mask is true.
///
/// Returns the count of non-zero elements in the mask.
///
/// # Safety
/// - `mask` must be valid pointer to `numel` u8 elements
#[inline]
pub unsafe fn masked_count_kernel(mask: *const u8, numel: usize) -> usize {
    let mask_slice = std::slice::from_raw_parts(mask, numel);
    mask_slice.iter().filter(|&&m| m != 0).count()
}

/// Select elements where mask is true, returning a flattened result.
///
/// # Arguments
/// * `a` - Input data pointer
/// * `mask` - Mask tensor pointer (u8: 0=false, non-zero=true)
/// * `out` - Output pointer (must be sized for count of true elements)
/// * `numel` - Number of elements in input/mask
///
/// # Safety
/// - All pointers must be valid for the specified size
/// - `out` must have enough space for all selected elements
#[inline]
pub unsafe fn masked_select_kernel<T: Element>(
    a: *const T,
    mask: *const u8,
    out: *mut T,
    numel: usize,
) {
    let a_slice = std::slice::from_raw_parts(a, numel);
    let mask_slice = std::slice::from_raw_parts(mask, numel);

    let mut out_idx = 0;
    for i in 0..numel {
        if mask_slice[i] != 0 {
            *out.add(out_idx) = a_slice[i];
            out_idx += 1;
        }
    }
}

/// Fill elements where mask is true with a scalar value.
///
/// # Arguments
/// * `a` - Input data pointer
/// * `mask` - Mask tensor pointer (u8: 0=false, non-zero=true)
/// * `out` - Output pointer
/// * `numel` - Number of elements
/// * `value` - Value to fill where mask is true
///
/// # Safety
/// - All pointers must be valid for the specified size
#[inline]
pub unsafe fn masked_fill_kernel<T: Element>(
    a: *const T,
    mask: *const u8,
    out: *mut T,
    numel: usize,
    value: f64,
) {
    let a_slice = std::slice::from_raw_parts(a, numel);
    let mask_slice = std::slice::from_raw_parts(mask, numel);
    let out_slice = std::slice::from_raw_parts_mut(out, numel);

    let fill_val = T::from_f64(value);

    for i in 0..numel {
        out_slice[i] = if mask_slice[i] != 0 {
            fill_val
        } else {
            a_slice[i]
        };
    }
}

/// Look up embeddings from an embedding table using indices.
///
/// This is the industry-standard embedding lookup operation used in neural networks
/// for word embeddings, entity embeddings, etc. Optimized for contiguous memory access.
///
/// # Algorithm
/// ```text
/// for i in 0..num_indices:
///     idx = indices[i]
///     if 0 <= idx < vocab_size:
///         output[i * embedding_dim..(i+1) * embedding_dim] = embeddings[idx * embedding_dim..(idx+1) * embedding_dim]
///     else:
///         output[i * embedding_dim..(i+1) * embedding_dim] = 0  // out of bounds
/// ```
///
/// # Arguments
/// * `embeddings` - 2D embedding table pointer [vocab_size, embedding_dim]
/// * `indices` - 1D/ND flattened index tensor pointer (i64 values)
/// * `out` - Output pointer [num_indices, embedding_dim]
/// * `num_indices` - Total number of indices (product of indices.shape())
/// * `vocab_size` - Size of vocabulary (embeddings.shape()[0])
/// * `embedding_dim` - Dimension of each embedding vector (embeddings.shape()[1])
///
/// # Safety
/// - All pointers must be valid for the specified sizes
/// - `out` must have space for `num_indices * embedding_dim` elements
///
/// # Performance
/// - Memory-bound operation - optimized for sequential reads of embedding rows
/// - Uses memcpy for efficient row copying when possible
/// - For large batches, consider using parallel version with Rayon
#[inline]
pub unsafe fn embedding_lookup_kernel<T: Element>(
    embeddings: *const T,
    indices: *const i64,
    out: *mut T,
    num_indices: usize,
    vocab_size: usize,
    embedding_dim: usize,
) {
    if num_indices == 0 || embedding_dim == 0 {
        return;
    }

    let indices_slice = std::slice::from_raw_parts(indices, num_indices);

    for (i, &idx_val) in indices_slice.iter().enumerate() {
        let out_offset = i * embedding_dim;

        // Check bounds
        if idx_val < 0 || idx_val as usize >= vocab_size {
            // Out of bounds - fill with zeros
            let out_slice = std::slice::from_raw_parts_mut(out.add(out_offset), embedding_dim);
            for elem in out_slice {
                *elem = T::zero();
            }
            continue;
        }

        let src_offset = (idx_val as usize) * embedding_dim;

        // Copy the entire embedding row (contiguous memory copy)
        std::ptr::copy_nonoverlapping(
            embeddings.add(src_offset),
            out.add(out_offset),
            embedding_dim,
        );
    }
}

/// Where (conditional select): out[i] = cond[i] ? x[i] : y[i]
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration
/// - AVX2: 8 f32s or 4 f64s per iteration
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Safety
/// - `cond` must be valid pointer to `len` u8 elements
/// - `x`, `y`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn where_kernel<T: Element>(
    cond: *const u8,
    x: *const T,
    y: *const T,
    out: *mut T,
    len: usize,
) {
    // Dispatch to SIMD for f32/f64 on x86-64
    #[cfg(target_arch = "x86_64")]
    {
        use super::simd::where_select;

        match T::DTYPE {
            DType::F32 => {
                where_select::where_f32(
                    cond,
                    x as *const f32,
                    y as *const f32,
                    out as *mut f32,
                    len,
                );
                return;
            }
            DType::F64 => {
                where_select::where_f64(
                    cond,
                    x as *const f64,
                    y as *const f64,
                    out as *mut f64,
                    len,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback
    where_kernel_scalar(cond, x, y, out, len);
}

/// Scalar where for all Element types
#[inline]
unsafe fn where_kernel_scalar<T: Element>(
    cond: *const u8,
    x: *const T,
    y: *const T,
    out: *mut T,
    len: usize,
) {
    let cond_slice = std::slice::from_raw_parts(cond, len);
    let x_slice = std::slice::from_raw_parts(x, len);
    let y_slice = std::slice::from_raw_parts(y, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        out_slice[i] = if cond_slice[i] != 0 {
            x_slice[i]
        } else {
            y_slice[i]
        };
    }
}

/// Where (conditional select) with broadcasting support
///
/// Uses strides to handle arbitrary broadcasting patterns. Stride of 0 means
/// the dimension is broadcast (all indices access the same element).
///
/// # Arguments
/// * `cond` - Pointer to condition tensor data (U8)
/// * `x` - Pointer to "true" values tensor data
/// * `y` - Pointer to "false" values tensor data
/// * `out` - Pointer to output tensor data
/// * `out_shape` - Shape of output tensor
/// * `cond_strides` - Strides for cond tensor (0 = broadcast dim)
/// * `x_strides` - Strides for x tensor (0 = broadcast dim)
/// * `y_strides` - Strides for y tensor (0 = broadcast dim)
/// * `cond_offset`, `x_offset`, `y_offset` - Starting offsets for each tensor
///
/// # Safety
/// - All pointers must be valid for the specified shapes and strides
/// - `out` must not overlap with input tensors
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn where_strided_kernel<T: Element>(
    cond: *const u8,
    x: *const T,
    y: *const T,
    out: *mut T,
    out_shape: &[usize],
    cond_strides: &[isize],
    x_strides: &[isize],
    y_strides: &[isize],
    cond_offset: usize,
    x_offset: usize,
    y_offset: usize,
) {
    let ndim = out_shape.len();
    let total = out_shape.iter().product::<usize>();

    if total == 0 {
        return;
    }

    // Optimize for common case: all inputs are contiguous and same shape
    let is_simple = ndim > 0 && {
        let mut expected_stride = 1isize;
        let mut simple = true;
        for i in (0..ndim).rev() {
            if cond_strides[i] != expected_stride
                || x_strides[i] != expected_stride
                || y_strides[i] != expected_stride
            {
                simple = false;
                break;
            }
            expected_stride *= out_shape[i] as isize;
        }
        simple && cond_offset == 0 && x_offset == 0 && y_offset == 0
    };

    if is_simple {
        // Fast path: use contiguous kernel
        where_kernel(cond, x, y, out, total);
        return;
    }

    // General strided iteration with incremental offset updates
    let mut indices = vec![0usize; ndim];
    let mut cond_idx = cond_offset as isize;
    let mut x_idx = x_offset as isize;
    let mut y_idx = y_offset as isize;

    for out_idx in 0..total {
        let cond_val = *cond.offset(cond_idx);
        let result = if cond_val != 0 {
            *x.offset(x_idx)
        } else {
            *y.offset(y_idx)
        };

        *out.add(out_idx) = result;

        // Increment multi-dimensional index with incremental offset updates
        for dim in (0..ndim).rev() {
            indices[dim] += 1;
            cond_idx += cond_strides[dim];
            x_idx += x_strides[dim];
            y_idx += y_strides[dim];

            if indices[dim] < out_shape[dim] {
                break;
            }

            // Reset this dimension and adjust offsets
            indices[dim] = 0;
            cond_idx -= (out_shape[dim] as isize) * cond_strides[dim];
            x_idx -= (out_shape[dim] as isize) * x_strides[dim];
            y_idx -= (out_shape[dim] as isize) * y_strides[dim];
        }
    }
}

/// Logical AND: out[i] = a[i] && b[i]
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` u8 elements
#[inline]
pub unsafe fn logical_and_kernel(a: *const u8, b: *const u8, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let b_slice = std::slice::from_raw_parts(b, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        out_slice[i] = if a_slice[i] != 0 && b_slice[i] != 0 {
            1
        } else {
            0
        };
    }
}

/// Logical OR: out[i] = a[i] || b[i]
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` u8 elements
#[inline]
pub unsafe fn logical_or_kernel(a: *const u8, b: *const u8, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let b_slice = std::slice::from_raw_parts(b, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        out_slice[i] = if a_slice[i] != 0 || b_slice[i] != 0 {
            1
        } else {
            0
        };
    }
}

/// Logical XOR: out[i] = a[i] ^ b[i]
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` u8 elements
#[inline]
pub unsafe fn logical_xor_kernel(a: *const u8, b: *const u8, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let b_slice = std::slice::from_raw_parts(b, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        // XOR: true if exactly one is true
        let a_bool = a_slice[i] != 0;
        let b_bool = b_slice[i] != 0;
        out_slice[i] = if a_bool != b_bool { 1 } else { 0 };
    }
}

/// Logical NOT: out[i] = !a[i]
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` u8 elements
#[inline]
pub unsafe fn logical_not_kernel(a: *const u8, out: *mut u8, len: usize) {
    let a_slice = std::slice::from_raw_parts(a, len);
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for i in 0..len {
        out_slice[i] = if a_slice[i] == 0 { 1 } else { 0 };
    }
}
