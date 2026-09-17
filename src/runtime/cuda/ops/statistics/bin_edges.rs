//! Bin edges tensor construction for CUDA histogram computation.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::TypeConversionOps;
use crate::runtime::common::statistics_common::compute_bin_edges_f64;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
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
            Tensor::<CudaRuntime>::from_slice(&edges_f32, &[bins + 1], &client.device)
        }
        DType::F64 => Tensor::<CudaRuntime>::from_slice(&edges_data, &[bins + 1], &client.device),
        _ => {
            // Create as F32 and cast
            let edges_f32: Vec<f32> = edges_data.iter().map(|&v| v as f32).collect();
            let edges = Tensor::<CudaRuntime>::from_slice(&edges_f32, &[bins + 1], &client.device)?;
            client.cast(&edges, dtype)
        }
    }
}
