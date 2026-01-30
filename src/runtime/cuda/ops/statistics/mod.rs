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
//!
//! # Native GPU Operations
//!
//! - **Mode**: Uses native CUDA kernel (`launch_mode_dim`) after GPU-based sorting.
//!   No CPU fallback - entire operation runs on GPU.

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
use crate::ops::TensorOps;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::runtime::statistics_common::compute_bin_edges_f64;
use crate::tensor::Tensor;

/// Create bin edges tensor from computed f64 edges.
pub(crate) fn create_bin_edges(
    client: &CudaClient,
    min_val: f64,
    max_val: f64,
    bins: usize,
    dtype: DType,
) -> Result<Tensor<CudaRuntime>> {
    let edges_data = compute_bin_edges_f64(min_val, max_val, bins);

    match dtype {
        DType::F32 => {
            let edges_f32: Vec<f32> = edges_data.iter().map(|&v| v as f32).collect();
            Ok(Tensor::<CudaRuntime>::from_slice(
                &edges_f32,
                &[bins + 1],
                &client.device,
            ))
        }
        DType::F64 => Ok(Tensor::<CudaRuntime>::from_slice(
            &edges_data,
            &[bins + 1],
            &client.device,
        )),
        _ => {
            // Create as F32 and cast
            let edges_f32: Vec<f32> = edges_data.iter().map(|&v| v as f32).collect();
            let edges = Tensor::<CudaRuntime>::from_slice(&edges_f32, &[bins + 1], &client.device);
            client.cast(&edges, dtype)
        }
    }
}

/// Extract scalar f64 value from tensor.
pub(crate) fn tensor_to_f64(t: &Tensor<CudaRuntime>) -> Result<f64> {
    let dtype = t.dtype();
    let t_contig = ensure_contiguous(t);

    let val = match dtype {
        DType::F32 => {
            let vec: Vec<f32> = t_contig.to_vec();
            vec[0] as f64
        }
        DType::F64 => {
            let vec: Vec<f64> = t_contig.to_vec();
            vec[0]
        }
        DType::I32 => {
            let vec: Vec<i32> = t_contig.to_vec();
            vec[0] as f64
        }
        DType::I64 => {
            let vec: Vec<i64> = t_contig.to_vec();
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
