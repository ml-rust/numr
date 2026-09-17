//! Statistics operations for WebGPU runtime
//!
//! Implements quantile, percentile, median, histogram, skewness, kurtosis, and mode
//! using native WGSL shaders where possible.
//!
//! This module shares common logic with other backends via the `statistics_common`
//! module to ensure consistency and reduce code duplication.
//!
//! # Module Organization
//!
//! - `quantile` - Quantile, percentile, and median operations
//! - `histogram` - Histogram computation
//! - `moments` - Skewness and kurtosis (higher-order moments)
//! - `mode` - Mode (most frequent value) using native WGSL shader
//!
//! # Native GPU Operations
//!
//! - **Mode**: Uses native WGSL shader (`launch_mode_dim`) after GPU-based sorting.
//!   No CPU fallback for supported dtypes (F32, I32, U32).

mod histogram;
mod mode;
mod moments;
mod quantile;
mod shared_helpers;

pub use histogram::histogram_impl;
pub use mode::mode_impl;
pub use moments::{kurtosis_impl, skew_impl};
pub use quantile::{median_impl, percentile_impl, quantile_impl};
