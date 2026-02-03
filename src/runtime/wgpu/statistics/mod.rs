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

pub use histogram::histogram_impl;
pub use mode::mode_impl;
pub use moments::{kurtosis_impl, skew_impl};
pub use quantile::{median_impl, percentile_impl, quantile_impl};

// ============================================================================
// Shared Helper Functions
// ============================================================================

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::TypeConversionOps;
use crate::runtime::statistics_common::compute_bin_edges_f64;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{RuntimeClient, ensure_contiguous};
use crate::tensor::Tensor;

/// Create bin edges tensor from computed f64 edges.
pub(crate) fn create_bin_edges(
    client: &WgpuClient,
    min_val: f64,
    max_val: f64,
    bins: usize,
    dtype: DType,
) -> Result<Tensor<WgpuRuntime>> {
    let edges_data = compute_bin_edges_f64(min_val, max_val, bins);

    // WebGPU primarily supports F32, so create as F32 first
    let edges_f32: Vec<f32> = edges_data.iter().map(|&v| v as f32).collect();
    let edges = Tensor::<WgpuRuntime>::from_slice(&edges_f32, &[bins + 1], client.device());

    if dtype == DType::F32 {
        Ok(edges)
    } else {
        client.cast(&edges, dtype)
    }
}

/// Extract scalar f64 value from tensor.
pub(crate) fn tensor_to_f64(t: &Tensor<WgpuRuntime>) -> Result<f64> {
    let dtype = t.dtype();
    let t_contig = ensure_contiguous(t);

    let val = match dtype {
        DType::F32 => {
            let vec: Vec<f32> = t_contig.to_vec();
            vec[0] as f64
        }
        DType::I32 => {
            let vec: Vec<i32> = t_contig.to_vec();
            vec[0] as f64
        }
        DType::U32 => {
            let vec: Vec<u32> = t_contig.to_vec();
            vec[0] as f64
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "tensor_to_f64",
            });
        }
    };

    Ok(val)
}
