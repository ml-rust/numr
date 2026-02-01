//! Normalization operations trait.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Normalization operations
pub trait NormalizationOps<R: Runtime> {
    /// RMS Normalization: output = input * rsqrt(mean(input^2) + eps) * weight
    ///
    /// RMSNorm is used in LLaMA and other modern transformer architectures.
    /// It normalizes over the last dimension.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [..., hidden_size]
    /// * `weight` - Weight tensor of shape [hidden_size]
    /// * `eps` - Small constant for numerical stability (typically 1e-5 or 1e-6)
    fn rms_norm(&self, input: &Tensor<R>, weight: &Tensor<R>, eps: f32) -> Result<Tensor<R>>;

    /// Layer Normalization: output = (input - mean) / sqrt(variance + eps) * weight + bias
    ///
    /// LayerNorm normalizes across the last dimension for each batch element.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [..., hidden_size]
    /// * `weight` - Weight (gamma) tensor of shape [hidden_size]
    /// * `bias` - Bias (beta) tensor of shape [hidden_size]
    /// * `eps` - Small constant for numerical stability (typically 1e-5)
    fn layer_norm(
        &self,
        input: &Tensor<R>,
        weight: &Tensor<R>,
        bias: &Tensor<R>,
        eps: f32,
    ) -> Result<Tensor<R>>;
}
