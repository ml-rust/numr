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
