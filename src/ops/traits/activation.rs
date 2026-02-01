//! Activation operations trait.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Activation operations
pub trait ActivationOps<R: Runtime> {
    /// Rectified linear unit: max(0, a)
    fn relu(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Sigmoid: 1 / (1 + e^(-a))
    fn sigmoid(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// SiLU (Swish): a * sigmoid(a) = a / (1 + e^(-a))
    ///
    /// Used in LLaMA, Mistral, and other modern transformer architectures.
    fn silu(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// GELU (Gaussian Error Linear Unit): 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3)))
    ///
    /// Uses the tanh approximation. Used in GPT, BERT, and other transformer architectures.
    fn gelu(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Leaky ReLU: max(negative_slope * a, a)
    ///
    /// Allows small gradients for negative inputs, helping prevent "dying ReLU" problem.
    /// Default negative_slope is typically 0.01.
    fn leaky_relu(&self, a: &Tensor<R>, negative_slope: f64) -> Result<Tensor<R>>;

    /// ELU (Exponential Linear Unit): a if a > 0, else alpha * (exp(a) - 1)
    ///
    /// Smooth approximation to ReLU with negative values saturating to -alpha.
    /// Default alpha is typically 1.0.
    fn elu(&self, a: &Tensor<R>, alpha: f64) -> Result<Tensor<R>>;

    /// Softmax along a dimension
    fn softmax(&self, a: &Tensor<R>, dim: isize) -> Result<Tensor<R>>;
}
