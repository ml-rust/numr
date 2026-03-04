//! Special mathematical functions for scientific computing.
//!
//! See [`traits`] for the `SpecialFunctions` trait, [`constants`] for
//! mathematical constants, and [`helpers`] for validation utilities.

pub mod bessel_coefficients;
pub mod constants;
pub mod helpers;
pub mod scalar;
pub mod traits;

pub use constants::{
    EULER_MASCHERONI, LANCZOS_COEFFICIENTS, LANCZOS_G, LN_SQRT_2PI, SQRT_PI, TWO_OVER_SQRT_PI,
};
pub use helpers::validate_special_dtype;
pub use scalar::*;
pub use traits::SpecialFunctions;
