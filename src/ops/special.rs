//! Special mathematical functions for scientific computing
//!
//! This module re-exports the `SpecialFunctions` trait from the algorithm module.
//! The trait provides error functions, gamma functions, beta functions, and
//! incomplete gamma/beta functions needed for probability distributions.
//!
//! # Usage
//!
//! ```ignore
//! use numr::ops::SpecialFunctions;
//!
//! let x = Tensor::from_slice(&[0.0, 0.5, 1.0], &[3], &device);
//! let erf_x = client.erf(&x)?;
//! ```
//!
//! # Available Functions
//!
//! - **Error functions**: `erf`, `erfc`, `erfinv`
//! - **Gamma functions**: `gamma`, `lgamma`, `digamma`
//! - **Beta functions**: `beta`, `betainc`
//! - **Incomplete gamma**: `gammainc`, `gammaincc`

// Re-export from algorithm module - single source of truth
pub use crate::algorithm::special::SpecialFunctions;
