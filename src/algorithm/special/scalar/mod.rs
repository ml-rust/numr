//! Scalar implementations of special mathematical functions
//!
//! This module provides high-precision scalar implementations that can be
//! applied element-wise by backend helper functions.
//!
//! # Algorithms
//!
//! - **erf/erfc**: Abramowitz & Stegun approximation 7.1.26 (~1e-7 accuracy)
//! - **erfinv**: Rational approximation + Newton-Raphson refinement
//! - **gamma/lgamma**: Lanczos approximation (g=7, n=9)
//! - **digamma**: Asymptotic expansion with recurrence relation
//! - **beta**: Computed via lgamma for numerical stability
//! - **betainc**: Continued fraction (Lentz's method)
//! - **betaincinv**: Newton's method with bisection fallback
//! - **gammainc/gammaincc**: Series expansion + continued fraction
//! - **gammaincinv**: Halley's method with asymptotic initial guess
//! - **bessel_***: Polynomial approximation (Numerical Recipes style)
//! - **ellipk/ellipe**: AGM method for elliptic integrals
//! - **hyp2f1/hyp1f1**: Series + transformations for hypergeometric functions
//! - **airy_ai/airy_bi**: Series + asymptotic for Airy functions
//! - **legendre_p/legendre_p_assoc**: Three-term recurrence for Legendre functions
//! - **sph_harm**: Spherical harmonics (real)
//! - **fresnel_s/fresnel_c**: Series + auxiliary functions

mod airy;
mod bessel;
mod elliptic;
mod error_functions;
mod fresnel;
pub mod gamma_functions;
mod hypergeometric;
mod inverse_functions;
mod orthogonal;

pub use airy::*;
pub use bessel::*;
pub use elliptic::*;
pub use error_functions::*;
pub use fresnel::*;
pub use gamma_functions::*;
pub use hypergeometric::*;
pub use inverse_functions::*;
pub use orthogonal::*;
