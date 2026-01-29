//! CPU implementation of special mathematical functions
//!
//! Implements error functions, gamma functions, beta functions, and incomplete
//! gamma/beta functions using well-established numerical algorithms.
//!
//! # Module Structure
//!
//! - `kernels` - Scalar computation functions (erf_scalar, gamma_scalar, etc.)
//! - `helpers` - Tensor operation adapters (apply_unary, apply_binary, etc.)

mod helpers;
mod kernels;

#[cfg(test)]
mod tests;

use crate::algorithm::special::{SpecialFunctions, validate_special_dtype};
use crate::error::Result;
use crate::tensor::Tensor;

use super::{CpuClient, CpuRuntime};

pub use kernels::*;

// ============================================================================
// SpecialFunctions Trait Implementation
// ============================================================================

impl SpecialFunctions<CpuRuntime> for CpuClient {
    fn erf(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, erf_scalar)
    }

    fn erfc(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, erfc_scalar)
    }

    fn erfinv(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, erfinv_scalar)
    }

    fn gamma(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, gamma_scalar)
    }

    fn lgamma(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, lgamma_scalar)
    }

    fn digamma(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, digamma_scalar)
    }

    fn beta(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        helpers::apply_binary(a, b, &self.device, beta_scalar)
    }

    fn betainc(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        helpers::apply_ternary(a, b, x, &self.device, betainc_scalar)
    }

    fn gammainc(
        &self,
        a: &Tensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        helpers::apply_binary(a, x, &self.device, gammainc_scalar)
    }

    fn gammaincc(
        &self,
        a: &Tensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        helpers::apply_binary(a, x, &self.device, gammaincc_scalar)
    }
}
