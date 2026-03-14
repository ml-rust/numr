//! Normalization operations trait.

use crate::error::{Error, Result};
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
    /// * `input` - Input tensor of shape `[..., hidden_size]`
    /// * `weight` - Weight tensor of shape `[hidden_size]`
    /// * `eps` - Small constant for numerical stability (typically 1e-5 or 1e-6)
    fn rms_norm(&self, input: &Tensor<R>, weight: &Tensor<R>, eps: f32) -> Result<Tensor<R>> {
        let _ = (input, weight, eps);
        Err(Error::NotImplemented {
            feature: "NormalizationOps::rms_norm",
        })
    }

    /// Layer Normalization: output = (input - mean) / sqrt(variance + eps) * weight + bias
    ///
    /// LayerNorm normalizes across the last dimension for each batch element.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape `[..., hidden_size]`
    /// * `weight` - Weight (gamma) tensor of shape `[hidden_size]`
    /// * `bias` - Bias (beta) tensor of shape `[hidden_size]`
    /// * `eps` - Small constant for numerical stability (typically 1e-5)
    fn layer_norm(
        &self,
        input: &Tensor<R>,
        weight: &Tensor<R>,
        bias: &Tensor<R>,
        eps: f32,
    ) -> Result<Tensor<R>> {
        let _ = (input, weight, bias, eps);
        Err(Error::NotImplemented {
            feature: "NormalizationOps::layer_norm",
        })
    }

    /// Group Normalization: normalize over groups of channels.
    ///
    /// Divides channels into `num_groups` groups and normalizes each group
    /// independently. Used in some vision architectures and diffusion models.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape `[batch, channels, ...]`
    /// * `weight` - Scale (gamma) of shape `[channels]`
    /// * `bias` - Bias (beta) of shape `[channels]`
    /// * `num_groups` - Number of groups (must divide channels evenly)
    /// * `eps` - Small constant for numerical stability
    fn group_norm(
        &self,
        input: &Tensor<R>,
        weight: &Tensor<R>,
        bias: &Tensor<R>,
        num_groups: usize,
        eps: f32,
    ) -> Result<Tensor<R>> {
        let _ = (input, weight, bias, num_groups, eps);
        Err(Error::NotImplemented {
            feature: "NormalizationOps::group_norm",
        })
    }

    /// Fused Add + RMS Normalization: pre_norm = x + residual, output = rms_norm(pre_norm, weight, eps)
    ///
    /// Saves one full memory pass vs separate add + rms_norm. Used in every
    /// transformer residual connection. Returns `(output, pre_norm)` where
    /// `pre_norm` is needed for backward pass and residual chaining.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[..., hidden_size]`
    /// * `residual` - Residual tensor of same shape as `x`
    /// * `weight` - Weight tensor of shape `[hidden_size]`
    /// * `eps` - Small constant for numerical stability
    fn fused_add_rms_norm(
        &self,
        x: &Tensor<R>,
        residual: &Tensor<R>,
        weight: &Tensor<R>,
        eps: f32,
    ) -> Result<(Tensor<R>, Tensor<R>)> {
        let _ = (x, residual, weight, eps);
        Err(Error::NotImplemented {
            feature: "NormalizationOps::fused_add_rms_norm",
        })
    }

    /// Backward pass for fused add + RMS normalization.
    ///
    /// Returns `(d_input_residual, d_weight)` where `d_input_residual` is the
    /// gradient for both `x` and `residual` (they share the same gradient since
    /// `d(x + residual)/dx = d(x + residual)/d(residual) = 1`).
    ///
    /// # Arguments
    ///
    /// * `grad` - Upstream gradient of shape `[..., hidden_size]`
    /// * `pre_norm` - The `x + residual` value from forward pass
    /// * `weight` - Weight tensor of shape `[hidden_size]`
    /// * `eps` - Same eps used in forward pass
    fn fused_add_rms_norm_bwd(
        &self,
        grad: &Tensor<R>,
        pre_norm: &Tensor<R>,
        weight: &Tensor<R>,
        eps: f32,
    ) -> Result<(Tensor<R>, Tensor<R>)> {
        let _ = (grad, pre_norm, weight, eps);
        Err(Error::NotImplemented {
            feature: "NormalizationOps::fused_add_rms_norm_bwd",
        })
    }

    /// Fused Add + Layer Normalization: pre_norm = x + residual, output = layer_norm(pre_norm, weight, bias, eps)
    ///
    /// Saves one full memory pass vs separate add + layer_norm.
    /// Returns `(output, pre_norm)`.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape `[..., hidden_size]`
    /// * `residual` - Residual tensor of same shape as `x`
    /// * `weight` - Weight (gamma) tensor of shape `[hidden_size]`
    /// * `bias` - Bias (beta) tensor of shape `[hidden_size]`
    /// * `eps` - Small constant for numerical stability
    fn fused_add_layer_norm(
        &self,
        x: &Tensor<R>,
        residual: &Tensor<R>,
        weight: &Tensor<R>,
        bias: &Tensor<R>,
        eps: f32,
    ) -> Result<(Tensor<R>, Tensor<R>)> {
        let _ = (x, residual, weight, bias, eps);
        Err(Error::NotImplemented {
            feature: "NormalizationOps::fused_add_layer_norm",
        })
    }

    /// Backward pass for fused add + layer normalization.
    ///
    /// Returns `(d_input_residual, d_weight, d_bias)`.
    ///
    /// # Arguments
    ///
    /// * `grad` - Upstream gradient of shape `[..., hidden_size]`
    /// * `pre_norm` - The `x + residual` value from forward pass
    /// * `weight` - Weight (gamma) tensor of shape `[hidden_size]`
    /// * `bias` - Bias (beta) tensor of shape `[hidden_size]`
    /// * `eps` - Same eps used in forward pass
    fn fused_add_layer_norm_bwd(
        &self,
        grad: &Tensor<R>,
        pre_norm: &Tensor<R>,
        weight: &Tensor<R>,
        bias: &Tensor<R>,
        eps: f32,
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)> {
        let _ = (grad, pre_norm, weight, bias, eps);
        Err(Error::NotImplemented {
            feature: "NormalizationOps::fused_add_layer_norm_bwd",
        })
    }
}
