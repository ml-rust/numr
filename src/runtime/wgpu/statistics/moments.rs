//! Higher-order moment statistics for WebGPU runtime (skewness, kurtosis)

use crate::error::Result;
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, StatisticalOps};
use crate::runtime::RuntimeClient;
use crate::runtime::statistics_common::DIVISION_EPSILON;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Compute skewness (third standardized moment) using composition.
///
/// Skewness measures the asymmetry of the distribution:
/// - Positive skew: right tail is longer
/// - Negative skew: left tail is longer
/// - Zero skew: symmetric distribution
pub fn skew_impl(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    // skew = E[(X - mean)^3] / std^3
    let mean = client.mean(a, dims, true)?;
    let centered = client.sub(a, &mean)?;

    // Compute third moment: mean((centered)^3)
    let centered_cubed = client.pow_scalar(&centered, 3.0)?;
    let m3 = client.mean(&centered_cubed, dims, keepdim)?;

    // Compute std^3
    let std_val = client.std(a, dims, keepdim, correction)?;
    let std_cubed = client.pow_scalar(&std_val, 3.0)?;

    // skew = m3 / std^3 (with epsilon to avoid division by zero)
    let epsilon = Tensor::<WgpuRuntime>::full_scalar(
        std_cubed.shape(),
        dtype,
        DIVISION_EPSILON,
        client.device(),
    );
    let std_cubed_safe = client.add(&std_cubed, &epsilon)?;

    client.div(&m3, &std_cubed_safe)
}

/// Compute kurtosis (fourth standardized moment, excess) using composition.
///
/// Excess kurtosis measures the "tailedness" of the distribution:
/// - Positive kurtosis: heavy tails (leptokurtic)
/// - Negative kurtosis: light tails (platykurtic)
/// - Zero kurtosis: normal distribution (mesokurtic)
pub fn kurtosis_impl(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    // kurtosis = E[(X - mean)^4] / std^4 - 3
    let mean = client.mean(a, dims, true)?;
    let centered = client.sub(a, &mean)?;

    // Compute fourth moment: mean((centered)^4)
    let centered_fourth = client.pow_scalar(&centered, 4.0)?;
    let m4 = client.mean(&centered_fourth, dims, keepdim)?;

    // Compute std^4
    let std_val = client.std(a, dims, keepdim, correction)?;
    let std_fourth = client.pow_scalar(&std_val, 4.0)?;

    // kurtosis = m4 / std^4 - 3 (with epsilon to avoid division by zero)
    let epsilon = Tensor::<WgpuRuntime>::full_scalar(
        std_fourth.shape(),
        dtype,
        DIVISION_EPSILON,
        client.device(),
    );
    let std_fourth_safe = client.add(&std_fourth, &epsilon)?;

    let ratio = client.div(&m4, &std_fourth_safe)?;
    let three = Tensor::<WgpuRuntime>::full_scalar(ratio.shape(), dtype, 3.0, client.device());
    client.sub(&ratio, &three)
}
