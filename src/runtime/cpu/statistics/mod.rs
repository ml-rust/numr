//! Statistics operations for CPU runtime
//!
//! Implements quantile, percentile, median, histogram, skewness, kurtosis, and mode.
//!
//! This module uses optimized CPU kernels with direct memory access for maximum
//! performance, while sharing common logic with other backends via the
//! `statistics_common` module.
//!
//! # Module Organization
//!
//! - `quantile` - Quantile, percentile, and median operations
//! - `histogram` - Histogram computation
//! - `moments` - Skewness and kurtosis (moment statistics)
//! - `mode` - Mode (most frequent value) computation
//! - `kernels` - Low-level unsafe quantile/histogram kernels
//! - `tensor_helpers` - Tensor construction and scalar-extraction helpers

mod histogram;
mod kernels;
mod mode;
mod moments;
mod quantile;
mod tensor_helpers;

// Re-export all public API functions
pub use histogram::histogram_impl;
pub use mode::mode_impl;
pub use moments::{kurtosis_impl, skew_impl};
pub use quantile::{median_impl, percentile_impl, quantile_impl};

// Re-export Interpolation for submodules
pub(crate) use crate::runtime::common::statistics_common::Interpolation;
