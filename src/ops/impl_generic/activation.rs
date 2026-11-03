//! Generic implementations of composite activation operations.

use crate::algorithm::special::SpecialFunctions;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::activation::normalize_softmax_dim;
use crate::ops::activation_common::validate_prelu;
use crate::ops::traits::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, CumulativeOps, RandomOps, ScalarOps,
    UnaryOps,
};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Generic softmax_with_bias implementation: `softmax(a + bias, dim)`.
///
/// This is the reference implementation used by CPU and any backend without a fused kernel.
/// Correctness is guaranteed by `softmax(broadcast_add(a, bias), dim)`.
pub fn softmax_with_bias_impl<R, C>(
    client: &C,
    a: &Tensor<R>,
    bias: &Tensor<R>,
    dim: isize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ActivationOps<R> + BinaryOps<R>,
{
    let biased = client.add(a, bias)?;
    client.softmax(&biased, dim)
}

/// Generic softplus implementation: softplus(x) = log(1 + exp(x))
///
/// Uses the numerically stable form: `relu(x) + log(1 + exp(-|x|))`
///
/// The naive formula `log(1 + exp(x))` overflows to `Inf` for large positive x
/// (e.g., x = 100: `exp(100) = Inf`). The stable decomposition keeps all
/// intermediate values bounded:
/// - For large x > 0: `relu(x) ≈ x`, `log(1 + exp(-x)) ≈ 0` → result ≈ x  ✓
/// - For large x < 0: `relu(x) = 0`, `log(1 + exp(-|x|)) ≈ exp(x)` → result ≈ exp(x)  ✓
/// - At x = 0: `0 + log(2) ≈ 0.693`  ✓
///
/// All backends delegate here — guarantees identical numerical behaviour.
pub fn softplus_impl<R, C>(client: &C, a: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ActivationOps<R> + UnaryOps<R> + ScalarOps<R> + BinaryOps<R>,
{
    // relu(x) = max(0, x)
    let relu_x = client.relu(a)?;

    // log(1 + exp(-|x|))  — all values bounded: exp(-|x|) ∈ (0, 1]
    let abs_x = client.abs(a)?;
    let neg_abs = client.neg(&abs_x)?;
    let exp_neg_abs = client.exp(&neg_abs)?;
    let one_plus = client.add_scalar(&exp_neg_abs, 1.0)?;
    let log_term = client.log(&one_plus)?;

    client.add(&relu_x, &log_term)
}

/// Standard normal CDF: `Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))`.
///
/// Shared by [`gelu_erf_impl`] (`gelu_erf(x) = x * Phi(x)`),
/// [`gelu_erf_mul_bwd_impl`] (`gelu_erf'(x) = Phi(x) + x * phi(x)`), and the
/// `var_gelu_erf_mul` autograd op, which precomputes `Phi` at forward time and
/// saves it on the `GradFn` so backward never needs an `erf` bound.
pub(crate) fn standard_normal_cdf<R, C>(client: &C, a: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: SpecialFunctions<R> + ScalarOps<R>,
{
    const FRAC_1_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let scaled = client.mul_scalar(a, FRAC_1_SQRT_2)?;
    let erf_val = client.erf(&scaled)?;
    let one_plus_erf = client.add_scalar(&erf_val, 1.0)?;
    client.mul_scalar(&one_plus_erf, 0.5)
}

/// Generic exact GELU implementation: `gelu_erf(x) = x * Phi(x)` where `Phi` is
/// the standard normal CDF, computed from the backend's native `erf`.
///
/// Matches PyTorch `F.gelu(x, approximate="none")`. Reuses whatever `erf`
/// implementation each backend already has (SIMD polynomial on CPU, native
/// `erff`/`erf` on CUDA, WGSL polynomial on WebGPU) instead of duplicating it.
pub fn gelu_erf_impl<R, C>(client: &C, a: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: SpecialFunctions<R> + ScalarOps<R> + BinaryOps<R>,
{
    let cdf = standard_normal_cdf(client, a)?;
    client.mul(a, &cdf)
}

/// Generic fused exact GELU-Mul: `gelu_erf(a) * b`.
pub fn gelu_erf_mul_impl<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: SpecialFunctions<R> + ScalarOps<R> + BinaryOps<R>,
{
    let cdf = standard_normal_cdf(client, a)?;
    let gelu_erf_a = client.mul(a, &cdf)?;
    client.mul(&gelu_erf_a, b)
}

/// Generic fused exact GELU-Mul backward: gradients for `output = gelu_erf(a) * b`.
///
/// `gelu_erf'(x) = Phi(x) + x * phi(x)` where `Phi` is the standard normal CDF
/// and `phi(x) = exp(-x^2 / 2) / sqrt(2*pi)` is the standard normal PDF.
pub fn gelu_erf_mul_bwd_impl<R, C>(
    client: &C,
    grad: &Tensor<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime,
    C: SpecialFunctions<R> + UnaryOps<R> + ScalarOps<R> + BinaryOps<R>,
{
    const INV_SQRT_2PI: f64 = 0.3989422804014327;

    let cdf = standard_normal_cdf(client, a)?;
    let gelu_erf_a = client.mul(a, &cdf)?;

    // phi(x) = exp(-x^2 / 2) / sqrt(2*pi)
    let neg_half_x_sq = client.mul_scalar(&client.mul(a, a)?, -0.5)?;
    let pdf = client.mul_scalar(&client.exp(&neg_half_x_sq)?, INV_SQRT_2PI)?;

    // gelu_erf'(x) = Phi(x) + x * phi(x)
    let x_pdf = client.mul(a, &pdf)?;
    let deriv = client.add(&cdf, &x_pdf)?;

    let d_b = client.mul(grad, &gelu_erf_a)?;
    let grad_times_b = client.mul(grad, b)?;
    let d_a = client.mul(&grad_times_b, &deriv)?;

    Ok((d_a, d_b))
}

/// Generic log_softmax implementation: log_softmax(x, dim) = x - logsumexp(x, dim, keepdim=true)
///
/// This is the canonical algorithm — all backends delegate here.
/// Numerically stable because logsumexp uses the max-subtraction trick internally.
pub fn log_softmax_impl<R, C>(client: &C, a: &Tensor<R>, dim: isize) -> Result<Tensor<R>>
where
    R: Runtime,
    C: BinaryOps<R> + CumulativeOps<R>,
{
    let ndim = a.ndim();
    let dim_idx = normalize_softmax_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

    let lse = client.logsumexp(a, &[dim_idx], true)?;
    client.sub(a, &lse)
}

/// Generic dropout implementation: where(rand > p, x / (1-p), 0)
///
/// During training, randomly zeros elements with probability `p` and scales
/// remaining elements by `1/(1-p)` to preserve expected values.
/// During inference (`training=false`), returns input unchanged.
pub fn dropout_impl<R, C>(client: &C, a: &Tensor<R>, p: f64, training: bool) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RandomOps<R> + CompareOps<R> + ConditionalOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    if !training || p == 0.0 {
        return Ok(a.clone());
    }
    if p >= 1.0 {
        return Ok(Tensor::<R>::zeros(a.shape(), a.dtype(), client.device())?);
    }

    // Generate random mask: rand > p means "keep"
    let rand_tensor = client.rand(a.shape(), a.dtype())?;
    let threshold = Tensor::<R>::full_scalar(a.shape(), a.dtype(), p, client.device())?;
    let mask = client.gt(&rand_tensor, &threshold)?;

    // Scale kept values by 1/(1-p)
    let scale = 1.0 / (1.0 - p);
    let scaled = client.mul_scalar(a, scale)?;

    // Zero out dropped elements
    let zeros = Tensor::<R>::zeros(a.shape(), a.dtype(), client.device())?;
    client.where_cond(&mask, &scaled, &zeros)
}

/// Generic PReLU implementation: `where(a > 0, a, weight * a)`.
///
/// `weight` must have the dtype of `a` and must broadcast to `a` without
/// changing `a`'s shape. All backends delegate here.
pub fn prelu_impl<R, C>(client: &C, a: &Tensor<R>, weight: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: CompareOps<R> + ConditionalOps<R> + BinaryOps<R> + RuntimeClient<R>,
{
    validate_prelu(a.dtype(), a.shape(), weight.dtype(), weight.shape())?;

    // A rank-0 zero broadcasts to any shape, so the mask needs no full-size zeros.
    let zero = Tensor::<R>::zeros(&[], a.dtype(), client.device())?;
    let positive = client.gt(a, &zero)?;
    let scaled = client.mul(a, weight)?;
    client.where_cond(&positive, a, &scaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::special::scalar::erf_scalar;
    use crate::ops::ActivationOps;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    /// Reference values: `0.5 * x * (1 + erf(x / sqrt(2)))`, computed from libm
    /// `erf` (double precision) — matches PyTorch `F.gelu(x, approximate="none")`
    /// to full `f64` precision.
    #[test]
    fn test_gelu_erf_matches_exact_gelu_f64() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let xs: [f64; 10] = [-8.0, -3.0, -1.0, -1e-3, 0.0, 1e-3, 0.5, 1.0, 3.0, 8.0];
        let expected: [f64; 10] = [
            -4.884_981_308_350_689e-15,
            -0.004_049_694_094_890_31,
            -0.158_655_253_931_457_07,
            -0.000_499_601_057_786_089,
            0.0,
            0.000_500_398_942_213_911,
            0.345_731_230_637_006_56,
            0.841_344_746_068_542_9,
            2.995_950_305_905_11,
            7.999_999_999_999_995,
        ];

        let a = Tensor::<CpuRuntime>::from_slice(&xs, &[xs.len()], &device).unwrap();
        let out = client.gelu_erf(&a).unwrap();
        let result: Vec<f64> = out.to_vec();

        for i in 0..xs.len() {
            let diff = (result[i] - expected[i]).abs();
            assert!(
                diff < 1e-12,
                "gelu_erf mismatch at x={}: got {}, want {}",
                xs[i],
                result[i],
                expected[i]
            );
        }
    }

    /// 137 elements: not a multiple of the AVX-512 (16), AVX2 (8), or NEON (4)
    /// f32 lane width used by the CPU `erf` kernel `gelu_erf` delegates to, so
    /// this exercises both the vectorized loop and the scalar tail.
    #[test]
    fn test_gelu_erf_large_batch_simd_lanes_and_tail() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let len = 137;
        let xs: Vec<f32> = (0..len).map(|i| (i as f32 - 68.0) * 0.25).collect();
        let a = Tensor::<CpuRuntime>::from_slice(&xs, &[len], &device).unwrap();
        let out = client.gelu_erf(&a).unwrap();
        let result: Vec<f32> = out.to_vec();

        for (i, &x) in xs.iter().enumerate() {
            let scaled = (x as f64) / std::f64::consts::SQRT_2;
            let want = 0.5 * (x as f64) * (1.0 + erf_scalar(scaled));
            let diff = (result[i] as f64 - want).abs();
            assert!(
                diff < 5e-6,
                "gelu_erf batch mismatch at i={i}, x={x}: got {}, want {want}",
                result[i]
            );
        }
    }

    fn cpu() -> (crate::runtime::cpu::CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        (CpuRuntime::default_client(&device), device)
    }

    fn prelu_f64(x: &[f64], x_shape: &[usize], w: &[f64], w_shape: &[usize]) -> Vec<f64> {
        let (client, device) = cpu();
        let a = Tensor::<CpuRuntime>::from_slice(x, x_shape, &device).unwrap();
        let weight = Tensor::<CpuRuntime>::from_slice(w, w_shape, &device).unwrap();
        let out = client.prelu(&a, &weight).unwrap();
        assert_eq!(out.shape(), x_shape);
        out.to_vec()
    }

    #[test]
    fn test_prelu_positive_negative_zero() {
        let out = prelu_f64(&[2.0, -2.0, 0.0, 0.5, -0.5], &[5], &[0.25], &[1]);
        assert_eq!(out, vec![2.0, -0.5, 0.0, 0.5, -0.125]);
    }

    #[test]
    fn test_prelu_per_channel_nchw() {
        // [N=1, C=2, H=1, W=2], weight [C, 1, 1].
        let out = prelu_f64(
            &[-1.0, 1.0, -2.0, 2.0],
            &[1, 2, 1, 2],
            &[0.5, 0.1],
            &[2, 1, 1],
        );
        assert_eq!(out, vec![-0.5, 1.0, -0.2, 2.0]);
    }

    #[test]
    fn test_prelu_per_channel_nhwc() {
        // [N=1, H=1, W=2, C=2], weight [C].
        let out = prelu_f64(&[-1.0, -2.0, 3.0, -4.0], &[1, 1, 2, 2], &[0.5, 0.25], &[2]);
        assert_eq!(out, vec![-0.5, -0.5, 3.0, -1.0]);
    }

    #[test]
    fn test_prelu_scalar_weight() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let a = Tensor::<CpuRuntime>::from_slice(&[-4.0f64, 4.0], &[2], &device).unwrap();
        let weight = Tensor::<CpuRuntime>::from_slice(&[0.5f64], &[], &device).unwrap();
        let out = client.prelu(&a, &weight).unwrap();
        assert_eq!(out.to_vec::<f64>(), vec![-2.0, 4.0]);
    }

    #[test]
    fn test_prelu_rejects_non_broadcasting_weight() {
        let (client, device) = cpu();
        let a = Tensor::<CpuRuntime>::from_slice(&[0.0f64; 6], &[2, 3], &device).unwrap();
        let weight = Tensor::<CpuRuntime>::from_slice(&[0.0f64; 4], &[4], &device).unwrap();
        let err = client.prelu(&a, &weight).unwrap_err();
        assert!(matches!(err, Error::BroadcastError { .. }));
        let msg = err.to_string();
        assert!(msg.contains("[2, 3]") && msg.contains("[4]"), "{msg}");
    }

    #[test]
    fn test_prelu_rejects_weight_that_grows_input() {
        // Broadcasts, but the output shape [3, 2] differs from the input shape [2].
        let (client, device) = cpu();
        let a = Tensor::<CpuRuntime>::from_slice(&[0.0f64; 2], &[2], &device).unwrap();
        let weight = Tensor::<CpuRuntime>::from_slice(&[0.0f64; 3], &[3, 1], &device).unwrap();
        let err = client.prelu(&a, &weight).unwrap_err();
        assert!(matches!(err, Error::BroadcastError { .. }));
    }

    #[test]
    fn test_prelu_rejects_dtype_mismatch() {
        let (client, device) = cpu();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -1.0], &[2], &device).unwrap();
        let weight = Tensor::<CpuRuntime>::from_slice(&[0.5f32], &[1], &device).unwrap();
        let err = client.prelu(&a, &weight).unwrap_err();
        assert!(matches!(err, Error::DTypeMismatch { .. }));
    }
}
