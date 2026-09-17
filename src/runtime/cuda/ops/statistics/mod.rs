//! Statistics operations for CUDA runtime
//!
//! Implements quantile, percentile, median, histogram, skewness, kurtosis, and mode
//! using native CUDA kernels where possible.
//!
//! This module shares common logic with other backends via the `statistics_common`
//! module to ensure consistency and reduce code duplication.
//!
//! # Module Organization
//!
//! - `quantile` - Quantile, percentile, and median operations
//! - `histogram` - Histogram computation
//! - `moments` - Skewness and kurtosis (higher-order moments)
//! - `mode` - Mode (most frequent value) using native CUDA kernel
//! - `bin_edges` - Bin edges tensor construction for histogram
//! - `scalar_read` - Single-scalar GPU-to-host readback
//!
//! # Native GPU Operations
//!
//! - **Mode**: Uses native CUDA kernel (`launch_mode_dim`) after GPU-based sorting.
//!   No CPU fallback - entire operation runs on GPU.

mod bin_edges;
mod histogram;
mod mode;
mod moments;
mod quantile;
mod scalar_read;

pub use histogram::histogram_impl;
pub use mode::mode_impl;
pub use moments::{kurtosis_impl, skew_impl};
pub use quantile::{median_impl, percentile_impl, quantile_impl};
