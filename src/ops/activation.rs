//! Activation function helpers
//!
//! This module contains helper types and functions for activation operations.
//! The actual operations are defined in the `TensorOps` trait.

/// Activation function kind
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ActivationKind {
    /// ReLU: max(0, x)
    ReLU,
    /// LeakyReLU: max(alpha * x, x) where alpha < 1
    LeakyReLU,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Tanh: hyperbolic tangent
    Tanh,
    /// GELU: Gaussian Error Linear Unit
    GELU,
    /// SiLU/Swish: x * sigmoid(x)
    SiLU,
    /// Softplus: log(1 + exp(x))
    Softplus,
    /// Mish: x * tanh(softplus(x))
    Mish,
}

/// Normalize softmax dimension (handle negative index)
pub fn normalize_softmax_dim(ndim: usize, dim: isize) -> Option<usize> {
    if dim >= 0 {
        let d = dim as usize;
        if d < ndim { Some(d) } else { None }
    } else {
        let d = ndim as isize + dim;
        if d >= 0 { Some(d as usize) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_softmax_dim() {
        assert_eq!(normalize_softmax_dim(3, 1), Some(1));
        assert_eq!(normalize_softmax_dim(3, -1), Some(2));
        assert_eq!(normalize_softmax_dim(3, 3), None);
        assert_eq!(normalize_softmax_dim(3, -4), None);
    }
}
