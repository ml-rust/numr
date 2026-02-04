//! Polynomial operations for numerical computing
//!
//! This module provides polynomial operations commonly used in signal processing,
//! control systems, and filter design. All operations use native numr primitives
//! and work across CPU, CUDA, and WebGPU backends.
//!
//! # Functions
//!
//! - [`PolynomialAlgorithms::polyroots`] - Find roots of a polynomial
//! - [`PolynomialAlgorithms::polyval`] - Evaluate polynomial at given points
//! - [`PolynomialAlgorithms::polyfromroots`] - Construct polynomial from roots
//! - [`PolynomialAlgorithms::polymul`] - Multiply two polynomials
//!
//! # Coefficient Convention
//!
//! Polynomials are represented as 1D tensors of coefficients in ascending order:
//! - `coeffs[0]` = constant term (c₀)
//! - `coeffs[n]` = leading coefficient (cₙ)
//! - Polynomial: p(x) = c₀ + c₁x + c₂x² + ... + cₙxⁿ
//!
//! This matches NumPy's polynomial coefficient ordering.
//!
//! # Complex Roots
//!
//! Roots are returned as separate real and imaginary tensors in [`PolynomialRoots`]
//! to support WebGPU which lacks native complex number support.

pub mod core;
pub mod helpers;
pub mod traits;
pub mod types;

pub use helpers::{
    validate_polynomial_coeffs, validate_polynomial_dtype, validate_polynomial_roots,
};
pub use traits::PolynomialAlgorithms;
pub use types::PolynomialRoots;
