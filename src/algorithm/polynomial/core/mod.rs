//! Core polynomial algorithms shared across all backends
//!
//! This module provides the unified implementation for polynomial operations.
//! All backends (CPU, CUDA, WebGPU) call these functions to ensure numerical parity.
//!
//! # Design
//!
//! These functions operate through the trait interface (LinearAlgebraAlgorithms,
//! BinaryOps, IndexingOps, etc.) using ONLY tensor operations. This ensures all
//! operations stay on the original device without GPU↔CPU transfers.
//!
//! # No GPU↔CPU Transfers
//!
//! All algorithms are implemented using tensor operations only:
//! - `index_select` for accessing individual coefficients
//! - `scatter_reduce` for convolution operations
//! - `arange` and `eye` for tensor construction
//! - Broadcasting for element-wise operations

mod convolve;
mod dtype_support;
mod index_helpers;
mod polyfromroots;
mod polymul;
mod polyroots;
mod polyval;

pub use convolve::convolve_impl;
pub use dtype_support::DTypeSupport;
pub use polyfromroots::polyfromroots_impl;
pub use polymul::polymul_impl;
pub use polyroots::polyroots_impl;
pub use polyval::polyval_impl;

pub(crate) use index_helpers::{create_arange_tensor, create_index_tensor};
