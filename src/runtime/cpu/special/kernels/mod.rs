//! Scalar implementations of special mathematical functions
//!
//! This module provides high-precision scalar implementations that are
//! applied element-wise by the helper functions in `helpers.rs`.
//!
//! # Algorithms
//!
//! - **erf/erfc**: Abramowitz & Stegun approximation 7.1.26 (~1e-7 accuracy)
//! - **erfinv**: Rational approximation + Newton-Raphson refinement
//! - **gamma/lgamma**: Lanczos approximation (g=7, n=9)
//! - **digamma**: Asymptotic expansion with recurrence relation
//! - **beta**: Computed via lgamma for numerical stability
//! - **betainc**: Continued fraction (Lentz's method)
//! - **gammainc/gammaincc**: Series expansion + continued fraction
//! - **bessel_***: Polynomial approximation (Numerical Recipes style)

mod bessel;
mod error_functions;
mod gamma_functions;

pub use bessel::*;
pub use error_functions::*;
pub use gamma_functions::*;
