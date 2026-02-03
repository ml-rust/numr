//! CPU implementation of special mathematical functions
//!
//! Implements error functions, gamma functions, beta functions, incomplete
//! gamma/beta functions, elliptic integrals, hypergeometric functions,
//! Airy functions, Legendre polynomials, and Fresnel integrals.
//!
//! # Module Structure
//!
//! - `algorithm::special::scalar` - Scalar computation functions (erf_scalar, gamma_scalar, etc.)
//! - `helpers` - Tensor operation adapters (apply_unary, apply_binary, etc.)

mod helpers;

#[cfg(test)]
mod tests;

use crate::algorithm::special::scalar::{
    airy_ai_scalar, airy_bi_scalar, bessel_k0_scalar, bessel_k1_scalar, bessel_y0_scalar,
    bessel_y1_scalar, beta_scalar, betainc_scalar, betaincinv_scalar, ellipe_scalar, ellipk_scalar,
    erfinv_scalar, fresnel_c_scalar, fresnel_s_scalar, gammainc_scalar, gammaincc_scalar,
    gammaincinv_scalar, hyp1f1_scalar, hyp2f1_scalar, legendre_p_assoc_scalar, legendre_p_scalar,
    sph_harm_scalar,
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
        helpers::apply_erf(x, &self.device)
    }

    fn erfc(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_erfc(x, &self.device)
    }

    fn erfinv(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, erfinv_scalar)
    }

    fn gamma(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_gamma(x, &self.device)
    }

    fn lgamma(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_lgamma(x, &self.device)
    }

    fn digamma(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_digamma(x, &self.device)
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
        helpers::apply_bessel_j0(x, &self.device)
    }

    fn bessel_j1(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_bessel_j1(x, &self.device)
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
        helpers::apply_bessel_i0(x, &self.device)
    }

    fn bessel_i1(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_bessel_i1(x, &self.device)
    }

    fn bessel_k0(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, bessel_k0_scalar)
    }

    fn bessel_k1(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, bessel_k1_scalar)
    }

    // ========================================================================
    // Extended Special Functions
    // ========================================================================

    fn ellipk(&self, m: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(m.dtype())?;
        helpers::apply_unary(m, &self.device, ellipk_scalar)
    }

    fn ellipe(&self, m: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(m.dtype())?;
        helpers::apply_unary(m, &self.device, ellipe_scalar)
    }

    fn hyp2f1(&self, a: f64, b: f64, c: f64, z: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(z.dtype())?;
        helpers::apply_unary_with_three_f64s(z, &self.device, a, b, c, hyp2f1_scalar)
    }

    fn hyp1f1(&self, a: f64, b: f64, z: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(z.dtype())?;
        helpers::apply_unary_with_two_f64s(z, &self.device, a, b, hyp1f1_scalar)
    }

    fn airy_ai(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, airy_ai_scalar)
    }

    fn airy_bi(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, airy_bi_scalar)
    }

    fn legendre_p(&self, n: i32, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary_with_int(x, &self.device, n, legendre_p_scalar)
    }

    fn legendre_p_assoc(
        &self,
        n: i32,
        m: i32,
        x: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary_with_two_ints(x, &self.device, n, m, legendre_p_assoc_scalar)
    }

    fn sph_harm(
        &self,
        n: i32,
        m: i32,
        theta: &Tensor<CpuRuntime>,
        phi: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(theta.dtype())?;
        helpers::apply_binary_with_two_ints(theta, phi, &self.device, n, m, sph_harm_scalar)
    }

    fn fresnel_s(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, fresnel_s_scalar)
    }

    fn fresnel_c(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        validate_special_dtype(x.dtype())?;
        helpers::apply_unary(x, &self.device, fresnel_c_scalar)
    }
}
