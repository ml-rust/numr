//! Index operation kernels (gather, scatter, masked operations)

use crate::dtype::Element;
use crate::ops::ScatterReduceOp;

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

/// Scatter values with reduction into a destination tensor.
///
/// # Arguments
/// * `dst` - Destination tensor data pointer
/// * `indices` - Index tensor pointer (i64 values)
/// * `src` - Source values to scatter
/// * `out` - Output pointer
/// * `counts` - Optional count buffer for Mean reduction (must be pre-zeroed)
/// * `shape` - Shape of destination tensor
/// * `index_shape` - Shape of index/src tensors
/// * `dim` - Dimension along which to scatter
/// * `op` - Reduction operation to apply
/// * `include_self` - Whether to include dst values in reduction
///
/// # Safety
/// - All pointers must be valid for the specified shapes
/// - `counts` must be valid if op == Mean and include_self == false
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn scatter_reduce_kernel<T: Element>(
    dst: *const T,
    indices: *const i64,
    src: *const T,
    out: *mut T,
    counts: *mut u32,
    shape: &[usize],
    index_shape: &[usize],
    dim: usize,
    op: ScatterReduceOp,
    include_self: bool,
) {
    let ndim = shape.len();
    if ndim == 0 {
        return;
    }

    let dst_numel: usize = shape.iter().product();

    // Initialize output based on operation and include_self
    if include_self {
        // Copy dst to out
        std::ptr::copy_nonoverlapping(dst, out, dst_numel);
        // Initialize counts to 1 for Mean
        if op == ScatterReduceOp::Mean && !counts.is_null() {
            let counts_slice = std::slice::from_raw_parts_mut(counts, dst_numel);
            for c in counts_slice.iter_mut() {
                *c = 1;
            }
        }
    } else {
        // Initialize based on reduction operation
        let out_slice = std::slice::from_raw_parts_mut(out, dst_numel);
        match op {
            ScatterReduceOp::Sum | ScatterReduceOp::Mean => {
                for elem in out_slice.iter_mut() {
                    *elem = T::zero();
                }
            }
            ScatterReduceOp::Prod => {
                for elem in out_slice.iter_mut() {
                    *elem = T::one();
                }
            }
            ScatterReduceOp::Max => {
                // Use negative infinity for Max initialization
                for elem in out_slice.iter_mut() {
                    *elem = T::from_f64(f64::NEG_INFINITY);
                }
            }
            ScatterReduceOp::Min => {
                // Use positive infinity for Min initialization
                for elem in out_slice.iter_mut() {
                    *elem = T::from_f64(f64::INFINITY);
                }
            }
        }
        // Initialize counts to 0 for Mean
        if op == ScatterReduceOp::Mean && !counts.is_null() {
            let counts_slice = std::slice::from_raw_parts_mut(counts, dst_numel);
            for c in counts_slice.iter_mut() {
                *c = 0;
            }
        }
    }

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

    // Scatter with reduction
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

        let src_val = *src.add(src_idx);
        let dst_val = *out.add(dst_offset);

        // Apply reduction operation
        let new_val = match op {
            ScatterReduceOp::Sum | ScatterReduceOp::Mean => dst_val + src_val,
            ScatterReduceOp::Prod => dst_val * src_val,
            ScatterReduceOp::Max => {
                if src_val.to_f64() > dst_val.to_f64() {
                    src_val
                } else {
                    dst_val
                }
            }
            ScatterReduceOp::Min => {
                if src_val.to_f64() < dst_val.to_f64() {
                    src_val
                } else {
                    dst_val
                }
            }
        };

        *out.add(dst_offset) = new_val;

        // Update count for Mean
        if op == ScatterReduceOp::Mean && !counts.is_null() {
            *counts.add(dst_offset) += 1;
        }
    }

    // Finalize Mean: divide by count
    if op == ScatterReduceOp::Mean && !counts.is_null() {
        let out_slice = std::slice::from_raw_parts_mut(out, dst_numel);
        let counts_slice = std::slice::from_raw_parts(counts, dst_numel);
        for (elem, &count) in out_slice.iter_mut().zip(counts_slice.iter()) {
            if count > 0 {
                *elem = T::from_f64(elem.to_f64() / count as f64);
            }
        }
    }
}

/// Gather elements using N-dimensional indices.
///
/// The last dimension of `indices` contains coordinates into `input`.
///
/// # Arguments
/// * `input` - Input data pointer
/// * `indices` - Index tensor pointer (i64 values)
/// * `out` - Output pointer
/// * `input_shape` - Shape of input tensor
/// * `indices_shape` - Shape of indices tensor
/// * `out_shape` - Shape of output tensor
///
/// # Safety
/// - All pointers must be valid for the specified shapes
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gather_nd_kernel<T: Element>(
    input: *const T,
    indices: *const i64,
    out: *mut T,
    input_shape: &[usize],
    indices_shape: &[usize],
    out_shape: &[usize],
) {
    if input_shape.is_empty() || indices_shape.is_empty() {
        return;
    }

    let input_ndim = input_shape.len();
    let indices_ndim = indices_shape.len();
    let index_depth = indices_shape[indices_ndim - 1]; // M: number of coordinates

    // Compute input strides
    let mut input_strides = vec![1usize; input_ndim];
    for i in (0..input_ndim - 1).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // Compute indices strides
    let mut indices_strides = vec![1usize; indices_ndim];
    for i in (0..indices_ndim - 1).rev() {
        indices_strides[i] = indices_strides[i + 1] * indices_shape[i + 1];
    }

    // Compute output strides
    let out_ndim = out_shape.len();
    let mut out_strides = vec![1usize; out_ndim.max(1)];
    for i in (0..out_ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // Number of index vectors (product of indices.shape[:-1])
    let num_indices: usize = indices_shape[..indices_ndim - 1]
        .iter()
        .product::<usize>()
        .max(1);

    // Size of trailing dimensions from input (after the indexed dimensions)
    let trailing_size: usize = if index_depth < input_ndim {
        input_shape[index_depth..].iter().product()
    } else {
        1
    };

    // For each index vector
    for idx_vec in 0..num_indices {
        // Compute offset into indices tensor for this index vector
        let indices_offset = idx_vec * index_depth;

        // Read the index coordinates
        let mut input_offset = 0usize;
        let mut valid = true;
        for d in 0..index_depth {
            let coord = *indices.add(indices_offset + d);
            if coord < 0 || coord as usize >= input_shape[d] {
                valid = false;
                break;
            }
            input_offset += (coord as usize) * input_strides[d];
        }

        // Compute output offset
        let out_offset = idx_vec * trailing_size;

        if !valid {
            // Out of bounds - fill with zeros
            for i in 0..trailing_size {
                *out.add(out_offset + i) = T::zero();
            }
        } else {
            // Copy trailing elements
            for i in 0..trailing_size {
                *out.add(out_offset + i) = *input.add(input_offset + i);
            }
        }
    }
}

/// Count occurrences of each value in an integer tensor.
///
/// # Arguments
/// * `input` - Input integer tensor pointer (i64 values)
/// * `weights` - Optional weights pointer (same length as input)
/// * `out` - Output pointer (histogram)
/// * `numel` - Number of elements in input
/// * `output_len` - Length of output histogram
///
/// # Safety
/// - All pointers must be valid for the specified sizes
/// - `input` values must be in range [0, output_len)
///
/// # Returns
/// * `true` if all values were non-negative, `false` if any negative value found
#[inline]
pub unsafe fn bincount_kernel<T: Element>(
    input: *const i64,
    weights: *const T,
    out: *mut T,
    numel: usize,
    output_len: usize,
) -> bool {
    // Initialize output to zero
    let out_slice = std::slice::from_raw_parts_mut(out, output_len);
    for elem in out_slice.iter_mut() {
        *elem = T::zero();
    }

    let input_slice = std::slice::from_raw_parts(input, numel);
    let has_weights = !weights.is_null();

    for i in 0..numel {
        let val = input_slice[i];
        if val < 0 {
            return false; // Negative value found
        }
        let idx = val as usize;
        if idx < output_len {
            if has_weights {
                let w = *weights.add(i);
                out_slice[idx] = out_slice[idx] + w;
            } else {
                out_slice[idx] = out_slice[idx] + T::one();
            }
        }
    }

    true
}

/// Find the maximum value in an i64 tensor.
///
/// # Safety
/// - `input` must be valid for `numel` elements
#[inline]
pub unsafe fn max_i64_kernel(input: *const i64, numel: usize) -> i64 {
    if numel == 0 {
        return -1;
    }
    let slice = std::slice::from_raw_parts(input, numel);
    *slice.iter().max().unwrap_or(&-1)
}
