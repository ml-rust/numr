//! Arithmetic operations helpers
//!
//! This module contains helper types and functions for arithmetic operations.
//! The actual operations are defined in the `TensorOps` trait.

use crate::tensor::Layout;

/// Compute the output shape for binary operations with broadcasting
///
/// Returns None if the shapes are incompatible for broadcasting.
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);

    // Iterate from right to left
    for i in 0..max_ndim {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if a_dim == b_dim {
            result.push(a_dim);
        } else if a_dim == 1 {
            result.push(b_dim);
        } else if b_dim == 1 {
            result.push(a_dim);
        } else {
            return None; // Incompatible shapes
        }
    }

    result.reverse();
    Some(result)
}

/// Check if two layouts are compatible for element-wise operations
///
/// Returns true if the shapes can be broadcast together.
pub fn can_broadcast(a: &Layout, b: &Layout) -> bool {
    broadcast_shape(a.shape(), b.shape()).is_some()
}

/// Binary operation kind
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    /// Addition: a + b
    Add,
    /// Subtraction: a - b
    Sub,
    /// Multiplication: a * b
    Mul,
    /// Division: a / b
    Div,
    /// Power: a^b
    Pow,
    /// Maximum: max(a, b)
    Max,
    /// Minimum: min(a, b)
    Min,
}

/// Unary operation kind
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    /// Negation: -a
    Neg,
    /// Absolute value: |a|
    Abs,
    /// Square root: sqrt(a)
    Sqrt,
    /// Exponential: e^a
    Exp,
    /// Natural log: ln(a)
    Log,
    /// Sine: sin(a)
    Sin,
    /// Cosine: cos(a)
    Cos,
    /// Tangent: tan(a)
    Tan,
    /// Hyperbolic tangent: tanh(a)
    Tanh,
    /// Reciprocal: 1/a
    Recip,
    /// Square: a^2
    Square,
    /// Floor: floor(a)
    Floor,
    /// Ceiling: ceil(a)
    Ceil,
    /// Round: round(a)
    Round,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shape() {
        // Same shapes
        assert_eq!(broadcast_shape(&[2, 3], &[2, 3]), Some(vec![2, 3]));

        // Broadcasting with 1
        assert_eq!(broadcast_shape(&[2, 3], &[1, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shape(&[2, 1], &[2, 3]), Some(vec![2, 3]));

        // Different ranks
        assert_eq!(broadcast_shape(&[3], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shape(&[2, 3], &[3]), Some(vec![2, 3]));

        // Incompatible shapes
        assert_eq!(broadcast_shape(&[2, 3], &[2, 4]), None);
        assert_eq!(broadcast_shape(&[3], &[4]), None);
    }
}
