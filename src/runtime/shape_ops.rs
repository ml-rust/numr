//! Shared shape operation utilities for all backends
//!
//! This module provides common validation and zero-copy implementations
//! for shape operations (cat, stack, split, chunk) that are identical
//! across all backends.
//!
//! # Design
//!
//! - Validation logic is implemented once here and used by all backends
//! - Zero-copy operations (split, chunk) are fully implemented here
//! - Copy operations (cat, stack) use validation from here, with backend-specific kernels

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

// ============================================================================
// Shared Utilities
// ============================================================================

/// Normalize a dimension index, supporting negative indexing.
///
/// Returns `None` if the dimension is out of bounds.
#[inline]
pub fn normalize_dim(dim: isize, ndim: usize) -> Option<usize> {
    if ndim == 0 {
        return None;
    }
    let idx = if dim < 0 {
        let adjusted = ndim as isize + dim;
        if adjusted < 0 {
            return None;
        }
        adjusted as usize
    } else {
        dim as usize
    };
    if idx < ndim { Some(idx) } else { None }
}

/// Normalize a dimension index for stack operation (allows ndim as valid index).
///
/// Stack inserts a new dimension, so valid range is 0..=ndim.
#[inline]
pub fn normalize_stack_dim(dim: isize, ndim: usize) -> Option<usize> {
    let new_ndim = ndim + 1;
    if dim < 0 {
        let adjusted = new_ndim as isize + dim;
        if adjusted < 0 || adjusted >= new_ndim as isize {
            None
        } else {
            Some(adjusted as usize)
        }
    } else {
        let idx = dim as usize;
        if idx <= ndim { Some(idx) } else { None }
    }
}

// ============================================================================
// Cat Validation
// ============================================================================

/// Parameters for cat operation after validation.
#[derive(Debug, Clone)]
pub struct CatParams {
    /// Normalized dimension index
    pub dim_idx: usize,
    /// Data type of all tensors
    pub dtype: DType,
    /// Total size along cat dimension
    pub cat_dim_total: usize,
    /// Output shape
    pub out_shape: Vec<usize>,
    /// Product of dimensions before cat dimension (>= 1)
    pub outer_size: usize,
    /// Product of dimensions after cat dimension (>= 1)
    pub inner_size: usize,
}

/// Validate inputs for cat operation and compute output parameters.
///
/// This is the single source of truth for cat validation, used by all backends.
pub fn validate_cat<R: Runtime>(tensors: &[&Tensor<R>], dim: isize) -> Result<CatParams> {
    // Validate: need at least one tensor
    if tensors.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "tensors",
            reason: "cat requires at least one tensor".to_string(),
        });
    }

    let first = tensors[0];
    let dtype = first.dtype();
    let ndim = first.ndim();

    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "tensors",
            reason: "cannot concatenate scalar tensors".to_string(),
        });
    }

    // Normalize dimension
    let dim_idx = normalize_dim(dim, ndim).ok_or(Error::InvalidDimension { dim, ndim })?;

    // Validate all tensors have same dtype and compatible shapes
    let mut cat_dim_total = first.shape()[dim_idx];
    for &tensor in &tensors[1..] {
        if tensor.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: tensor.dtype(),
            });
        }
        if tensor.ndim() != ndim {
            return Err(Error::ShapeMismatch {
                expected: first.shape().to_vec(),
                got: tensor.shape().to_vec(),
            });
        }
        // Check all dims match except cat dimension
        for (i, (&a, &b)) in first.shape().iter().zip(tensor.shape().iter()).enumerate() {
            if i != dim_idx && a != b {
                return Err(Error::ShapeMismatch {
                    expected: first.shape().to_vec(),
                    got: tensor.shape().to_vec(),
                });
            }
        }
        cat_dim_total += tensor.shape()[dim_idx];
    }

    // Compute output shape
    let mut out_shape = first.shape().to_vec();
    out_shape[dim_idx] = cat_dim_total;

    // Compute outer/inner sizes for the cat algorithm
    let outer_size: usize = out_shape[..dim_idx].iter().product();
    let outer_size = outer_size.max(1);
    let inner_size: usize = out_shape[dim_idx + 1..].iter().product();
    let inner_size = inner_size.max(1);

    Ok(CatParams {
        dim_idx,
        dtype,
        cat_dim_total,
        out_shape,
        outer_size,
        inner_size,
    })
}

// ============================================================================
// Stack Validation
// ============================================================================

/// Validate inputs for stack operation.
///
/// Returns the normalized dimension index for the new dimension.
pub fn validate_stack<R: Runtime>(tensors: &[&Tensor<R>], dim: isize) -> Result<usize> {
    // Validate: need at least one tensor
    if tensors.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "tensors",
            reason: "stack requires at least one tensor".to_string(),
        });
    }

    let first = tensors[0];
    let dtype = first.dtype();
    let ndim = first.ndim();

    // For stack, dim can be 0..=ndim (we're inserting a new dimension)
    let dim_idx = normalize_stack_dim(dim, ndim).ok_or(Error::InvalidDimension {
        dim,
        ndim: ndim + 1,
    })?;

    // Validate all tensors have same dtype and exact same shape
    for &tensor in &tensors[1..] {
        if tensor.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: tensor.dtype(),
            });
        }
        if tensor.shape() != first.shape() {
            return Err(Error::ShapeMismatch {
                expected: first.shape().to_vec(),
                got: tensor.shape().to_vec(),
            });
        }
    }

    Ok(dim_idx)
}

// ============================================================================
// Split/Chunk (Zero-Copy, Backend-Agnostic)
// ============================================================================

/// Split a tensor into chunks of a given size along a dimension.
///
/// This is a zero-copy operation that returns views into the original tensor.
/// Works identically across all backends.
pub fn split_impl<R: Runtime>(
    tensor: &Tensor<R>,
    split_size: usize,
    dim: isize,
) -> Result<Vec<Tensor<R>>> {
    if split_size == 0 {
        return Err(Error::InvalidArgument {
            arg: "split_size",
            reason: "split_size must be greater than zero".to_string(),
        });
    }

    let ndim = tensor.ndim();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: "cannot split a scalar tensor".to_string(),
        });
    }

    let dim_idx = normalize_dim(dim, ndim).ok_or(Error::InvalidDimension { dim, ndim })?;
    let dim_size = tensor.shape()[dim_idx];

    let mut result = Vec::new();
    let mut start = 0;

    while start < dim_size {
        let length = (dim_size - start).min(split_size);
        let chunk = tensor.narrow(dim, start, length)?;
        result.push(chunk);
        start += length;
    }

    Ok(result)
}

/// Split a tensor into a specific number of chunks along a dimension.
///
/// This is a zero-copy operation that returns views into the original tensor.
/// Works identically across all backends.
pub fn chunk_impl<R: Runtime>(
    tensor: &Tensor<R>,
    chunks: usize,
    dim: isize,
) -> Result<Vec<Tensor<R>>> {
    if chunks == 0 {
        return Err(Error::InvalidArgument {
            arg: "chunks",
            reason: "chunks must be greater than zero".to_string(),
        });
    }

    let ndim = tensor.ndim();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: "cannot chunk a scalar tensor".to_string(),
        });
    }

    let dim_idx = normalize_dim(dim, ndim).ok_or(Error::InvalidDimension { dim, ndim })?;
    let dim_size = tensor.shape()[dim_idx];

    // Calculate chunk sizes: earlier chunks may be one larger if not evenly divisible
    let base_size = dim_size / chunks;
    let remainder = dim_size % chunks;

    let mut result = Vec::new();
    let mut start = 0;

    for i in 0..chunks {
        // First `remainder` chunks get one extra element
        let length = if i < remainder {
            base_size + 1
        } else {
            base_size
        };
        if length > 0 {
            let chunk = tensor.narrow(dim, start, length)?;
            result.push(chunk);
            start += length;
        }
    }

    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_dim_positive() {
        assert_eq!(normalize_dim(0, 3), Some(0));
        assert_eq!(normalize_dim(1, 3), Some(1));
        assert_eq!(normalize_dim(2, 3), Some(2));
        assert_eq!(normalize_dim(3, 3), None); // out of bounds
    }

    #[test]
    fn test_normalize_dim_negative() {
        assert_eq!(normalize_dim(-1, 3), Some(2));
        assert_eq!(normalize_dim(-2, 3), Some(1));
        assert_eq!(normalize_dim(-3, 3), Some(0));
        assert_eq!(normalize_dim(-4, 3), None); // out of bounds
    }

    #[test]
    fn test_normalize_dim_zero_ndim() {
        assert_eq!(normalize_dim(0, 0), None);
        assert_eq!(normalize_dim(-1, 0), None);
    }

    #[test]
    fn test_normalize_stack_dim() {
        // For ndim=2, valid stack dims are 0, 1, 2 (new_ndim=3)
        assert_eq!(normalize_stack_dim(0, 2), Some(0));
        assert_eq!(normalize_stack_dim(1, 2), Some(1));
        assert_eq!(normalize_stack_dim(2, 2), Some(2));
        assert_eq!(normalize_stack_dim(3, 2), None); // out of bounds

        // Negative indexing
        assert_eq!(normalize_stack_dim(-1, 2), Some(2));
        assert_eq!(normalize_stack_dim(-2, 2), Some(1));
        assert_eq!(normalize_stack_dim(-3, 2), Some(0));
        assert_eq!(normalize_stack_dim(-4, 2), None); // out of bounds
    }
}
