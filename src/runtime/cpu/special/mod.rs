//! CPU implementation of special mathematical functions
//!
//! Implements error functions, gamma functions, beta functions, and incomplete
//! gamma/beta functions using well-established numerical algorithms.
//!
//! # Module Structure
//!
//! - `algorithm::special::scalar` - Scalar computation functions (erf_scalar, gamma_scalar, etc.)
//! - `helpers` - Tensor operation adapters (apply_unary, apply_binary, etc.)

mod helpers;

#[cfg(test)]
mod tests;

use crate::algorithm::special::scalar::{
    bessel_i0_scalar, bessel_i1_scalar, bessel_j0_scalar, bessel_j1_scalar, bessel_k0_scalar,
    bessel_k1_scalar, bessel_y0_scalar, bessel_y1_scalar, beta_scalar, betainc_scalar,
    betaincinv_scalar, digamma_scalar, erf_scalar, erfc_scalar, erfinv_scalar, gamma_scalar,
    gammainc_scalar, gammaincc_scalar, gammaincinv_scalar, lgamma_scalar,
};
use crate::algorithm::special::{SpecialFunctions, validate_special_dtype};
use crate::error::Result;
use crate::tensor::Tensor;

use super::{CpuClient, CpuRuntime};

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

    fn gammaincinv(
        &self,
        a: &Tensor<CpuRuntime>,
        p: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        helpers::apply_binary(a, p, &self.device, gammaincinv_scalar)
    }

    fn betaincinv(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        p: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(a.dtype())?;
        helpers::apply_ternary(a, b, p, &self.device, betaincinv_scalar)
    }

    fn bessel_j0(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, bessel_j0_scalar)
    }

    fn bessel_j1(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, bessel_j1_scalar)
    }

    fn bessel_y0(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, bessel_y0_scalar)
    }

    fn bessel_y1(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, bessel_y1_scalar)
    }

    fn bessel_i0(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, bessel_i0_scalar)
    }

    fn bessel_i1(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, bessel_i1_scalar)
    }

    fn bessel_k0(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, bessel_k0_scalar)
    }

    fn bessel_k1(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, bessel_k1_scalar)
    }
}
