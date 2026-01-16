//! Reduction operations helpers
//!
//! This module contains helper types and functions for reduction operations.
//! The actual operations are defined in the `TensorOps` trait.

/// Reduction operation kind
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum of elements
    Sum,
    /// Mean of elements
    Mean,
    /// Maximum element
    Max,
    /// Minimum element
    Min,
    /// Product of elements
    Prod,
    /// Logical AND (for bool tensors)
    All,
    /// Logical OR (for bool tensors)
    Any,
}

/// Accumulation precision for reduction operations.
///
/// Controls the intermediate precision used during reduction for reduced-precision types:
/// - F16/BF16: Can use Native, FP32, or FP64 (default: Native)
/// - FP8: Can use BF16, FP32, or FP64 (default: FP32) - no native FP8 arithmetic
/// - F32: Can use Native or FP64 (default: Native)
/// - F64/integers: Always use native precision
///
/// # Memory vs Precision Trade-off
///
/// | Precision | Memory per element | Use case |
/// |-----------|-------------------|----------|
/// | Native | dtype size | Default, least memory |
/// | BF16 | 2 bytes | FP8 with moderate precision |
/// | FP32 | 4 bytes | Good numerical stability |
/// | FP64 | 8 bytes | Maximum precision (math/science) |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccumulationPrecision {
    /// Use native dtype for accumulation.
    /// Least memory usage, may have reduced precision for large reductions.
    /// For FP8, this is equivalent to FP32 (no native FP8 arithmetic).
    #[default]
    Native,
    /// Use BF16 for accumulation (for FP8 types).
    /// Uses less shared memory than FP32 (2 bytes vs 4 bytes per element).
    /// For F16/BF16, this is equivalent to Native or FP32 respectively.
    BF16,
    /// Use FP32 for accumulation.
    /// Good numerical stability for large reductions.
    /// Uses 4 bytes per element.
    FP32,
    /// Use FP64 for accumulation.
    /// Maximum precision for math/science applications.
    /// Uses 8 bytes per element.
    FP64,
}

/// Compute output shape for reduction
///
/// # Arguments
/// * `input_shape` - Shape of input tensor
/// * `dims` - Dimensions to reduce over
/// * `keepdim` - If true, keep reduced dimensions as size 1
pub fn reduce_output_shape(input_shape: &[usize], dims: &[usize], keepdim: bool) -> Vec<usize> {
    if keepdim {
        // Keep all dimensions, set reduced ones to 1
        input_shape
            .iter()
            .enumerate()
            .map(|(i, &s)| if dims.contains(&i) { 1 } else { s })
            .collect()
    } else {
        // Remove reduced dimensions
        input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !dims.contains(i))
            .map(|(_, &s)| s)
            .collect()
    }
}

/// Normalize reduction dimensions (handle negative indices)
///
/// Returns None if any dimension is out of range.
pub fn normalize_dims(ndim: usize, dims: &[isize]) -> Option<Vec<usize>> {
    dims.iter()
        .map(|&d| {
            if d >= 0 {
                let d = d as usize;
                if d < ndim { Some(d) } else { None }
            } else {
                let d = ndim as isize + d;
                if d >= 0 { Some(d as usize) } else { None }
            }
        })
        .collect()
}

/// All dimensions for full reduction
pub fn all_dims(ndim: usize) -> Vec<usize> {
    (0..ndim).collect()
}

/// Compute the strides for a single-dimension reduction (used by argmax/argmin).
///
/// Returns `(outer_size, reduce_size, inner_size)` where:
/// - `outer_size`: product of dimensions before the reduced dimension
/// - `reduce_size`: size of the dimension being reduced
/// - `inner_size`: product of dimensions after the reduced dimension
///
/// This is the standard decomposition for implementing reduce operations that
/// iterate over outer × inner combinations, each reducing over reduce_size elements.
///
/// # Arguments
/// * `shape` - Shape of the input tensor
/// * `dim` - The dimension to reduce over
///
/// # Example
/// ```ignore
/// let shape = &[2, 3, 4];
/// let (outer, reduce, inner) = compute_reduce_strides(shape, 1);
/// assert_eq!((outer, reduce, inner), (2, 3, 4));
/// ```
#[inline]
pub fn compute_reduce_strides(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer_size: usize = shape[..dim].iter().product::<usize>().max(1);
    let reduce_size = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product::<usize>().max(1);
    (outer_size, reduce_size, inner_size)
}

/// Compute output shape for a single-dimension reduction (used by argmax/argmin).
///
/// This is a convenience wrapper around [`reduce_output_shape`] for the common
/// case of reducing over exactly one dimension.
///
/// # Arguments
/// * `shape` - Shape of the input tensor
/// * `dim` - The dimension to reduce over
/// * `keepdim` - If true, keep the reduced dimension as size 1
#[inline]
pub fn reduce_dim_output_shape(shape: &[usize], dim: usize, keepdim: bool) -> Vec<usize> {
    reduce_output_shape(shape, &[dim], keepdim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_output_shape() {
        // Reduce single dim without keepdim
        assert_eq!(reduce_output_shape(&[2, 3, 4], &[1], false), vec![2, 4]);

        // Reduce single dim with keepdim
        assert_eq!(reduce_output_shape(&[2, 3, 4], &[1], true), vec![2, 1, 4]);

        // Reduce multiple dims
        assert_eq!(reduce_output_shape(&[2, 3, 4], &[0, 2], false), vec![3]);
        assert_eq!(
            reduce_output_shape(&[2, 3, 4], &[0, 2], true),
            vec![1, 3, 1]
        );

        // Reduce all dims
        assert_eq!(reduce_output_shape(&[2, 3, 4], &[0, 1, 2], false), vec![]);
        assert_eq!(
            reduce_output_shape(&[2, 3, 4], &[0, 1, 2], true),
            vec![1, 1, 1]
        );
    }

    #[test]
    fn test_normalize_dims() {
        // Positive dims
        assert_eq!(normalize_dims(3, &[0, 1]), Some(vec![0, 1]));

        // Negative dims
        assert_eq!(normalize_dims(3, &[-1]), Some(vec![2]));
        assert_eq!(normalize_dims(3, &[-2, -1]), Some(vec![1, 2]));

        // Out of range
        assert_eq!(normalize_dims(3, &[3]), None);
        assert_eq!(normalize_dims(3, &[-4]), None);
    }
}
