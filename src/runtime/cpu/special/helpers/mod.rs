//! Tensor operation adapters for special functions
//!
//! This module provides helpers for applying special functions to tensors:
//! - `scalar`: Generic apply_* functions for element-wise operations
//! - `simd`: SIMD-optimized wrappers for key special functions

pub mod scalar;
pub mod simd;

// Re-export scalar helpers
pub use scalar::{
    apply_binary, apply_binary_with_two_ints, apply_ternary, apply_unary, apply_unary_with_int,
    apply_unary_with_three_f64s, apply_unary_with_two_f64s, apply_unary_with_two_ints,
};

// Re-export SIMD-optimized functions
pub use simd::{
    apply_bessel_i0, apply_bessel_i1, apply_bessel_j0, apply_bessel_j1, apply_digamma, apply_erf,
    apply_erfc, apply_gamma, apply_lgamma,
};
