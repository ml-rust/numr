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
mod special;
