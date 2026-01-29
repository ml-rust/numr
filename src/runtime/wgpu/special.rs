//! WebGPU implementation of special mathematical functions
//!
//! This module implements the [`SpecialFunctions`] trait for WebGPU using
//! native WGSL compute shaders.
//!
//! # Supported Functions
//!
//! Unary: erf, erfc, erfinv, gamma, lgamma, digamma
//! Binary: beta, gammainc, gammaincc
//! Ternary: betainc
//!
//! # Numerical Algorithms
//!
//! - erf/erfc: Abramowitz & Stegun 7.1.26 rational approximation
//! - erfinv: Newton-Raphson iteration
//! - gamma/lgamma: Lanczos approximation (g=7, n=9)
//! - digamma: Asymptotic expansion + recurrence relation
//! - beta: exp(lgamma(a) + lgamma(b) - lgamma(a+b))
//! - betainc: Continued fraction expansion (Lentz's method)
//! - gammainc/gammaincc: Power series and continued fractions

use super::WgpuRuntime;
use super::client::WgpuClient;
use super::shaders::{launch_special_binary, launch_special_ternary, launch_special_unary};
use crate::algorithm::special::{SpecialFunctions, validate_special_dtype};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

// Use helpers from ops module
use super::ops::helpers::{alloc_output, get_tensor_buffer};

impl SpecialFunctions<WgpuRuntime> for WgpuClient {
    fn erf(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "erf")
    }

    fn erfc(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "erfc")
    }

    fn erfinv(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "erfinv")
    }

    fn gamma(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "gamma")
    }

    fn lgamma(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "lgamma")
    }

    fn digamma(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        compute_unary_special(self, x, "digamma")
    }

    fn beta(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }
        compute_binary_special(self, a, b, "beta")
    }

    fn betainc(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != b.dtype() || a.dtype() != x.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        if a.shape() != b.shape() || a.shape() != x.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: x.shape().to_vec(),
            });
        }
        compute_ternary_special(self, a, b, x, "betainc")
    }

    fn gammainc(
        &self,
        a: &Tensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != x.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: x.dtype(),
            });
        }
        if a.shape() != x.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: x.shape().to_vec(),
            });
        }
        compute_binary_special(self, a, x, "gammainc")
    }

    fn gammaincc(
        &self,
        a: &Tensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        if a.dtype() != x.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: x.dtype(),
            });
        }
        if a.shape() != x.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: x.shape().to_vec(),
            });
        }
        compute_binary_special(self, a, x, "gammaincc")
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute unary special function using native WGSL shader
fn compute_unary_special(
    client: &WgpuClient,
    x: &Tensor<WgpuRuntime>,
    op: &str,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = x.dtype();

    // WebGPU only supports F32 for special functions (no F64)
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WebGPU special function (only F32 supported)",
        });
    }

    let shape = x.shape().to_vec();
    let numel = x.numel();

    // Get input buffer and allocate output
    let input_buffer = get_tensor_buffer(x)?;
    let output = alloc_output(client, &shape, dtype);
    let output_buffer = get_tensor_buffer(&output)?;

    // Launch native WGSL shader
    launch_special_unary(
        client.pipeline_cache(),
        &client.queue,
        op,
        &input_buffer,
        &output_buffer,
        numel as u32,
        dtype,
    )?;

    Ok(output)
}

/// Compute binary special function using native WGSL shader
fn compute_binary_special(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    op: &str,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    // WebGPU only supports F32 for special functions (no F64)
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WebGPU special function (only F32 supported)",
        });
    }

    let shape = a.shape().to_vec();
    let numel = a.numel();

    // Get input buffers and allocate output
    let input_a_buffer = get_tensor_buffer(a)?;
    let input_b_buffer = get_tensor_buffer(b)?;
    let output = alloc_output(client, &shape, dtype);
    let output_buffer = get_tensor_buffer(&output)?;

    // Launch native WGSL shader
    launch_special_binary(
        client.pipeline_cache(),
        &client.queue,
        op,
        &input_a_buffer,
        &input_b_buffer,
        &output_buffer,
        numel as u32,
        dtype,
    )?;

    Ok(output)
}

/// Compute ternary special function using native WGSL shader
fn compute_ternary_special(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    x: &Tensor<WgpuRuntime>,
    op: &str,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    // WebGPU only supports F32 for special functions (no F64)
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WebGPU special function (only F32 supported)",
        });
    }

    let shape = a.shape().to_vec();
    let numel = a.numel();

    // Get input buffers and allocate output
    let input_a_buffer = get_tensor_buffer(a)?;
    let input_b_buffer = get_tensor_buffer(b)?;
    let input_x_buffer = get_tensor_buffer(x)?;
    let output = alloc_output(client, &shape, dtype);
    let output_buffer = get_tensor_buffer(&output)?;

    // Launch native WGSL shader
    launch_special_ternary(
        client.pipeline_cache(),
        &client.queue,
        op,
        &input_a_buffer,
        &input_b_buffer,
        &input_x_buffer,
        &output_buffer,
        numel as u32,
        dtype,
    )?;

    Ok(output)
}
