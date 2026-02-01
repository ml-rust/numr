//! Histogram operations for WebGPU runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{TensorOps, TypeConversionOps};
use crate::runtime::statistics_common::compute_histogram_counts;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{RuntimeClient, ensure_contiguous};
use crate::tensor::Tensor;

use super::{create_bin_edges, tensor_to_f64};

/// Compute histogram of values using composition.
///
/// # Implementation Notes
///
/// Histogram counting is performed on CPU due to WebGPU's limited atomic support.
/// The bin edges are created on GPU.
pub fn histogram_impl(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    bins: usize,
    range: Option<(f64, f64)>,
) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
    if bins == 0 {
        return Err(Error::InvalidArgument {
            arg: "bins",
            reason: "Number of bins must be positive".to_string(),
        });
    }

    let dtype = a.dtype();
    let numel = a.numel();

    if numel == 0 {
        let (min_val, max_val) = range.unwrap_or((0.0, 1.0));
        let hist = Tensor::<WgpuRuntime>::zeros(&[bins], DType::I64, client.device());
        let edges = create_bin_edges(client, min_val, max_val, bins, dtype)?;
        return Ok((hist, edges));
    }

    // Flatten input
    let flat = a.reshape(&[numel])?;

    // Determine range
    let (min_val, max_val) = if let Some((min, max)) = range {
        if min >= max {
            return Err(Error::InvalidArgument {
                arg: "range",
                reason: format!("Range min ({}) must be less than max ({})", min, max),
            });
        }
        (min, max)
    } else {
        let min_tensor = client.min(&flat, &[], false)?;
        let max_tensor = client.max(&flat, &[], false)?;
        let min_val = tensor_to_f64(&min_tensor)?;
        let max_val = tensor_to_f64(&max_tensor)?;

        // Handle case where all values are the same
        if (min_val - max_val).abs() < f64::EPSILON {
            (min_val - 0.5, max_val + 0.5)
        } else {
            (min_val, max_val)
        }
    };

    // Ensure flat is contiguous for to_vec
    let flat_contig = ensure_contiguous(&flat);

    // Copy to CPU for histogram counting using shared implementation
    let counts = match dtype {
        DType::F32 => {
            let data: Vec<f32> = flat_contig.to_vec();
            compute_histogram_counts(&data, bins, min_val, max_val)
        }
        _ => {
            // Cast to F32 for processing
            let flat_f32 = client.cast(&flat, DType::F32)?;
            let flat_f32_contig = ensure_contiguous(&flat_f32);
            let data: Vec<f32> = flat_f32_contig.to_vec();
            compute_histogram_counts(&data, bins, min_val, max_val)
        }
    };

    // Copy counts to GPU
    let hist = Tensor::<WgpuRuntime>::from_slice(&counts, &[bins], client.device());

    // Create bin edges
    let edges = create_bin_edges(client, min_val, max_val, bins, dtype)?;

    Ok((hist, edges))
}
