//! Activation operations trait.

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Activation operations
pub trait ActivationOps<R: Runtime> {
    /// Rectified linear unit: max(0, a)
    fn relu(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "ActivationOps::relu",
        })
    }

    /// Sigmoid: 1 / (1 + e^(-a))
    fn sigmoid(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "ActivationOps::sigmoid",
        })
    }

    /// SiLU (Swish): a * sigmoid(a) = a / (1 + e^(-a))
    ///
    /// Used in LLaMA, Mistral, and other modern transformer architectures.
    fn silu(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "ActivationOps::silu",
        })
    }

    /// GELU (Gaussian Error Linear Unit): 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3)))
    ///
    /// Uses the tanh approximation. Used in GPT, BERT, and other transformer architectures.
    fn gelu(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "ActivationOps::gelu",
        })
    }

    /// Leaky ReLU: max(negative_slope * a, a)
    ///
    /// Allows small gradients for negative inputs, helping prevent "dying ReLU" problem.
    /// Default negative_slope is typically 0.01.
    fn leaky_relu(&self, a: &Tensor<R>, negative_slope: f64) -> Result<Tensor<R>> {
        let _ = (a, negative_slope);
        Err(Error::NotImplemented {
            feature: "ActivationOps::leaky_relu",
        })
    }

    /// ELU (Exponential Linear Unit): a if a > 0, else alpha * (exp(a) - 1)
    ///
    /// Smooth approximation to ReLU with negative values saturating to -alpha.
    /// Default alpha is typically 1.0.
    fn elu(&self, a: &Tensor<R>, alpha: f64) -> Result<Tensor<R>> {
        let _ = (a, alpha);
        Err(Error::NotImplemented {
            feature: "ActivationOps::elu",
        })
    }

    /// Softmax along a dimension
    fn softmax(&self, a: &Tensor<R>, dim: isize) -> Result<Tensor<R>> {
        let _ = (a, dim);
        Err(Error::NotImplemented {
            feature: "ActivationOps::softmax",
        })
    }

    /// Log-softmax along a dimension: log(softmax(x, dim))
    ///
    /// Computed as `x - logsumexp(x, dim)` for numerical stability.
    /// Used in log-probability calculations, Bayesian inference,
    /// categorical distributions, and information theory.
    fn log_softmax(&self, a: &Tensor<R>, dim: isize) -> Result<Tensor<R>> {
        let _ = (a, dim);
        Err(Error::NotImplemented {
            feature: "ActivationOps::log_softmax",
        })
    }

    /// Softmax backward pass: computes gradient w.r.t. input given output gradient and softmax output.
    ///
    /// Formula: `d_input = output * (grad - sum(grad * output, dim, keepdim=true))`
    ///
    /// This is the Jacobian-vector product for softmax, used in training backward passes.
    ///
    /// # Arguments
    /// * `grad` - Upstream gradient (same shape as output)
    /// * `output` - The softmax output from the forward pass
    /// * `dim` - The dimension along which softmax was computed
    fn softmax_bwd(&self, grad: &Tensor<R>, output: &Tensor<R>, dim: isize) -> Result<Tensor<R>> {
        let _ = (grad, output, dim);
        Err(Error::NotImplemented {
            feature: "ActivationOps::softmax_bwd",
        })
    }

    /// Softplus: `log(1 + exp(a))`
    ///
    /// A smooth approximation to ReLU that is always positive and differentiable.
    /// Used in Mamba2 for dt (step size) processing via `softplus(dt_proj(x)) + dt_bias`.
    ///
    /// Gradient: `sigmoid(a)`
    fn softplus(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "ActivationOps::softplus",
        })
    }

    /// Fused SiLU-Mul: `silu(a) * b` in a single pass.
    ///
    /// Computes `(a / (1 + exp(-a))) * b` element-wise with one memory pass
    /// instead of two (activation + multiply). Used in SwiGLU and similar gated architectures.
    fn silu_mul(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (a, b);
        Err(Error::NotImplemented {
            feature: "ActivationOps::silu_mul",
        })
    }

    /// Fused GELU-Mul: `gelu(a) * b` in a single pass.
    ///
    /// Computes `(0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715*a^3)))) * b` element-wise.
    /// Used in GeGLU gated architectures.
    fn gelu_mul(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (a, b);
        Err(Error::NotImplemented {
            feature: "ActivationOps::gelu_mul",
        })
    }

    /// Fused ReLU-Mul: `relu(a) * b` in a single pass.
    ///
    /// Computes `max(0, a) * b` element-wise. Used in ReGLU gated architectures.
    fn relu_mul(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (a, b);
        Err(Error::NotImplemented {
            feature: "ActivationOps::relu_mul",
        })
    }

    /// Fused Sigmoid-Mul: `sigmoid(a) * b` in a single pass.
    ///
    /// Computes `(1 / (1 + exp(-a))) * b` element-wise. Used in SiGLU gated architectures.
    fn sigmoid_mul(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (a, b);
        Err(Error::NotImplemented {
            feature: "ActivationOps::sigmoid_mul",
        })
    }

    /// Fused SiLU-Mul backward: computes gradients for `output = silu(a) * b`.
    ///
    /// Returns `(d_a, d_b)` where:
    /// - `d_a = grad * b * silu'(a)` with `silu'(x) = sigmoid(x) * (1 + x - silu(x))`
    /// - `d_b = grad * silu(a)`
    ///
    /// Backends may implement this as a single fused kernel for better performance.
    fn silu_mul_bwd(
        &self,
        grad: &Tensor<R>,
        a: &Tensor<R>,
        b: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>)> {
        let _ = (grad, a, b);
        Err(Error::NotImplemented {
            feature: "ActivationOps::silu_mul_bwd",
        })
    }

    /// Fused GELU-Mul backward: computes gradients for `output = gelu(a) * b`.
    ///
    /// Returns `(d_a, d_b)` where:
    /// - `d_a = grad * b * gelu'(a)`
    /// - `d_b = grad * gelu(a)`
    fn gelu_mul_bwd(
        &self,
        grad: &Tensor<R>,
        a: &Tensor<R>,
        b: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>)> {
        let _ = (grad, a, b);
        Err(Error::NotImplemented {
            feature: "ActivationOps::gelu_mul_bwd",
        })
    }

    /// Fused ReLU-Mul backward: computes gradients for `output = relu(a) * b`.
    ///
    /// Returns `(d_a, d_b)` where:
    /// - `d_a = grad * b * relu'(a)` with `relu'(x) = 1 if x > 0, else 0`
    /// - `d_b = grad * relu(a)`
    fn relu_mul_bwd(
        &self,
        grad: &Tensor<R>,
        a: &Tensor<R>,
        b: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>)> {
        let _ = (grad, a, b);
        Err(Error::NotImplemented {
            feature: "ActivationOps::relu_mul_bwd",
        })
    }

    /// Fused Sigmoid-Mul backward: computes gradients for `output = sigmoid(a) * b`.
    ///
    /// Returns `(d_a, d_b)` where:
    /// - `d_a = grad * b * sigmoid'(a)` with `sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))`
    /// - `d_b = grad * sigmoid(a)`
    fn sigmoid_mul_bwd(
        &self,
        grad: &Tensor<R>,
        a: &Tensor<R>,
        b: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>)> {
        let _ = (grad, a, b);
        Err(Error::NotImplemented {
            feature: "ActivationOps::sigmoid_mul_bwd",
        })
    }

    /// Fused softmax with additive bias: `softmax(a + bias, dim)` in a single memory pass.
    ///
    /// Semantically equivalent to `softmax(a + bias, dim)` but avoids materializing the
    /// intermediate `a + bias` tensor. The bias is added element-wise (with broadcasting)
    /// during the max/exp/sum scan so that only one pass over `a` is needed.
    ///
    /// # Arguments
    /// * `a` - Input tensor (e.g. `[B, H, S, S]` attention scores)
    /// * `bias` - Additive bias broadcastable to `a` (e.g. `[B, 1, 1, S]` attention mask)
    /// * `dim` - Dimension along which softmax is computed
    ///
    /// # Semantics
    /// Must produce results numerically identical (within FP tolerance) to
    /// `softmax(add_broadcast(a, bias), dim)`. Accumulation is always done in F32 even
    /// when `a` and `bias` are F16/BF16.
    fn softmax_with_bias(&self, a: &Tensor<R>, bias: &Tensor<R>, dim: isize) -> Result<Tensor<R>> {
        let _ = (a, bias, dim);
        Err(Error::NotImplemented {
            feature: "ActivationOps::softmax_with_bias",
        })
    }

    /// Dropout: randomly zero elements with probability `p` during training.
    ///
    /// When `training` is true, each element is independently zeroed with probability `p`,
    /// and remaining elements are scaled by `1/(1-p)` to maintain expected values.
    /// When `training` is false, returns the input unchanged.
    ///
    /// Used in regularization, Monte Carlo dropout, and Bayesian approximation.
    fn dropout(&self, a: &Tensor<R>, p: f64, training: bool) -> Result<Tensor<R>> {
        let _ = (a, p, training);
        Err(Error::NotImplemented {
            feature: "ActivationOps::dropout",
        })
    }
}
