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

/// Softmax parameters
#[derive(Copy, Clone, Debug)]
pub struct SoftmaxParams {
    /// Dimension to apply softmax over
    pub dim: usize,
    /// Whether to compute log-softmax (numerically stable log(softmax(x)))
    pub log: bool,
}

impl SoftmaxParams {
    /// Create softmax params for a given dimension
    pub fn new(dim: usize) -> Self {
        Self { dim, log: false }
    }

    /// Create log-softmax params
    pub fn log_softmax(dim: usize) -> Self {
        Self { dim, log: true }
    }
}

/// Normalize softmax dimension (handle negative index)
pub fn normalize_softmax_dim(ndim: usize, dim: isize) -> Option<usize> {
    if dim >= 0 {
        let d = dim as usize;
        if d < ndim {
            Some(d)
        } else {
            None
        }
    } else {
        let d = ndim as isize + dim;
        if d >= 0 {
            Some(d as usize)
        } else {
            None
        }
    }
}

/// Constants for GELU approximation
pub mod gelu {
    /// GELU coefficient: sqrt(2/pi)
    pub const SQRT_2_OVER_PI: f64 = 0.7978845608028654;

    /// GELU coefficient for tanh approximation
    pub const TANH_COEF: f64 = 0.044715;

    /// Compute GELU(x) using the tanh approximation
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    #[inline]
    pub fn gelu_tanh_approx(x: f64) -> f64 {
        0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + TANH_COEF * x * x * x)).tanh())
    }
}

/// Constants for SiLU/Swish
pub mod silu {
    /// Compute SiLU(x) = x * sigmoid(x)
    #[inline]
    pub fn silu(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
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

    #[test]
    fn test_gelu_approx() {
        // GELU(0) = 0
        assert!((gelu::gelu_tanh_approx(0.0) - 0.0).abs() < 1e-10);

        // GELU is approximately x for large positive x
        let x = 5.0;
        assert!((gelu::gelu_tanh_approx(x) - x).abs() < 0.01);

        // GELU is approximately 0 for large negative x
        assert!((gelu::gelu_tanh_approx(-5.0)).abs() < 0.01);
    }

    #[test]
    fn test_silu() {
        // SiLU(0) = 0
        assert!((silu::silu(0.0) - 0.0).abs() < 1e-10);

        // SiLU is approximately x for large positive x
        let x = 5.0;
        assert!((silu::silu(x) - x).abs() < 0.05);
    }
}
