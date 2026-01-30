//! Advanced decomposition algorithms: rsf2csf, QZ, and polar decomposition
//!
//! This module contains implementations for advanced matrix decompositions:
//! - rsf2csf: Convert Real Schur Form to Complex Schur Form
//! - qz: Generalized Schur (QZ) decomposition
//! - polar: Polar decomposition A = U @ P

mod polar;
mod qz;
mod rsf2csf;

pub use polar::polar_decompose_impl;
pub use qz::qz_decompose_impl;
pub use rsf2csf::rsf2csf_impl;

#[cfg(test)]
mod tests;
