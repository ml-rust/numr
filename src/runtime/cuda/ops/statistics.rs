//! Statistics operations for CUDA runtime
//!
//! Implements quantile, percentile, median, histogram, skewness, and kurtosis
//! using composition of existing CUDA operations.
//!
//! This module shares common logic with other backends via the `statistics_common`
//! module to ensure consistency and reduce code duplication.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{ScalarOps, TensorOps, compute_reduce_strides, reduce_dim_output_shape};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::statistics_common::{
    DIVISION_EPSILON, Interpolation, compute_bin_edges_f64, compute_histogram_counts,
    compute_quantile_interpolation,
};
use crate::runtime::{ensure_contiguous, normalize_dim};
use crate::tensor::Tensor;

// Re-export Interpolation for backward compatibility
#[allow(unused_imports)]
pub use crate::runtime::statistics_common::Interpolation as InterpolationMethod;

// ============================================================================
// Public API Functions
// ============================================================================

/// Compute quantile along a dimension using composition.
///
/// # Algorithm
///
/// 1. Sort along dimension (using existing CUDA sort)
/// 2. Copy sorted data to CPU
/// 3. Perform interpolation on CPU
/// 4. Copy result back to GPU
///
/// This hybrid approach is efficient because:
/// - Sorting on GPU is fast (O(n log n))
/// - Interpolation output is small (reduced dimension)
/// - Avoids custom CUDA kernel complexity
///
/// # Arguments
///
/// * `client` - The CUDA runtime client
/// * `a` - Input tensor
/// * `q` - Quantile to compute, must be in [0.0, 1.0]
/// * `dim` - Dimension to reduce along (None = flatten first)
/// * `keepdim` - Whether to keep the reduced dimension
/// * `interpolation` - Interpolation method ("linear", "lower", "higher", "nearest", "midpoint")
///
/// # Returns
///
/// Tensor containing the quantile values.
pub fn quantile_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    q: f64,
    dim: Option<isize>,
    keepdim: bool,
    interpolation: &str,
) -> Result<Tensor<CudaRuntime>> {
    // Validate q is in [0, 1]
    if !(0.0..=1.0).contains(&q) {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("Quantile q must be in [0, 1], got {}", q),
        });
    }

    let interp = Interpolation::parse(interpolation)?;
    let dtype = a.dtype();

    // Handle None dim: flatten to 1D first
    if dim.is_none() {
        let numel = a.numel();
        if numel == 0 {
            let out_shape = if keepdim { vec![1; a.ndim()] } else { vec![] };
            return Ok(Tensor::<CudaRuntime>::empty(
                &out_shape,
                dtype,
                &client.device,
            ));
        }

        let flat = a.reshape(&[numel])?;
        return quantile_impl(client, &flat, q, Some(0), keepdim, interpolation);
    }

    let dim_val = dim.unwrap();
    let shape = a.shape();
    let ndim = shape.len();

    if ndim == 0 {
        return Ok(a.clone());
    }

    let dim_idx = normalize_dim(dim_val, ndim)?;
    let dim_size = shape[dim_idx];

    if dim_size == 0 {
        let out_shape = reduce_dim_output_shape(shape, dim_idx, keepdim);
        return Ok(Tensor::<CudaRuntime>::empty(
            &out_shape,
            dtype,
            &client.device,
        ));
    }

    // Sort along dimension using CUDA sort
    let sorted = client.sort(a, dim_val, false)?;

    // Compute output shape and strides
    let out_shape = reduce_dim_output_shape(shape, dim_idx, keepdim);
    let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim_idx);

    // Calculate quantile indices using shared logic
    let (floor_idx, ceil_idx, frac) =
        crate::runtime::statistics_common::compute_quantile_indices(q, reduce_size);

    // Ensure sorted is contiguous for data access
    let sorted_contig = ensure_contiguous(&sorted);

    // Check for empty output
    let out_numel = out_shape.iter().product::<usize>();
    if out_numel == 0 {
        return Ok(Tensor::<CudaRuntime>::empty(
            &out_shape,
            dtype,
            &client.device,
        ));
    }

    // Compute on CPU and copy back using shared implementation
    match dtype {
        DType::F32 => {
            let sorted_data: Vec<f32> = sorted_contig.to_vec();
            let result = compute_quantile_interpolation(
                &sorted_data,
                outer_size,
                reduce_size,
                inner_size,
                floor_idx,
                ceil_idx,
                frac,
                interp,
            );
            Ok(Tensor::<CudaRuntime>::from_slice(
                &result,
                &out_shape,
                &client.device,
            ))
        }
        DType::F64 => {
            let sorted_data: Vec<f64> = sorted_contig.to_vec();
            let result = compute_quantile_interpolation(
                &sorted_data,
                outer_size,
                reduce_size,
                inner_size,
                floor_idx,
                ceil_idx,
                frac,
                interp,
            );
            Ok(Tensor::<CudaRuntime>::from_slice(
                &result,
                &out_shape,
                &client.device,
            ))
        }
        _ => {
            // For other dtypes, cast to f32, compute, cast back
            let sorted_f32 = client.cast(&sorted, DType::F32)?;
            let result_f32 = quantile_impl(
                client,
                &sorted_f32,
                q,
                Some(dim_val),
                keepdim,
                interpolation,
            )?;
            client.cast(&result_f32, dtype)
        }
    }
}

/// Compute percentile (quantile * 100) along a dimension.
///
/// # Arguments
///
/// * `client` - The CUDA runtime client
/// * `a` - Input tensor
/// * `p` - Percentile to compute, must be in [0.0, 100.0]
/// * `dim` - Dimension to reduce along (None = flatten first)
/// * `keepdim` - Whether to keep the reduced dimension
///
/// # Returns
///
/// Tensor containing the percentile values.
pub fn percentile_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    p: f64,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<Tensor<CudaRuntime>> {
    if !(0.0..=100.0).contains(&p) {
        return Err(Error::InvalidArgument {
            arg: "p",
            reason: format!("Percentile p must be in [0, 100], got {}", p),
        });
    }

    quantile_impl(client, a, p / 100.0, dim, keepdim, "linear")
}

/// Compute median (50th percentile) along a dimension.
///
/// # Arguments
///
/// * `client` - The CUDA runtime client
/// * `a` - Input tensor
/// * `dim` - Dimension to reduce along (None = flatten first)
/// * `keepdim` - Whether to keep the reduced dimension
///
/// # Returns
///
/// Tensor containing the median values.
pub fn median_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<Tensor<CudaRuntime>> {
    quantile_impl(client, a, 0.5, dim, keepdim, "linear")
}

/// Compute histogram of values using composition.
///
/// # Implementation Notes
///
/// Uses GPU for min/max computation, CPU for bin counting.
/// This avoids atomic operations on GPU which would require a custom kernel.
///
/// # Arguments
///
/// * `client` - The CUDA runtime client
/// * `a` - Input tensor (will be flattened)
/// * `bins` - Number of histogram bins (must be > 0)
/// * `range` - Optional (min, max) range; defaults to (a.min(), a.max())
///
/// # Returns
///
/// Tuple of (histogram counts as I64 tensor, bin edges tensor).
pub fn histogram_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    bins: usize,
    range: Option<(f64, f64)>,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
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
        let hist = Tensor::<CudaRuntime>::zeros(&[bins], DType::I64, &client.device);
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
        DType::F64 => {
            let data: Vec<f64> = flat_contig.to_vec();
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
    let hist = Tensor::<CudaRuntime>::from_slice(&counts, &[bins], &client.device);

    // Create bin edges
    let edges = create_bin_edges(client, min_val, max_val, bins, dtype)?;

    Ok((hist, edges))
}

/// Compute skewness (third standardized moment) using composition.
///
/// # Arguments
///
/// * `client` - The CUDA runtime client
/// * `a` - Input tensor
/// * `dims` - Dimensions to reduce (empty = all)
/// * `keepdim` - Whether to keep reduced dimensions
/// * `correction` - Bias correction (reserved for future use)
///
/// # Returns
///
/// Tensor containing the skewness values.
pub fn skew_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();

    // skew = E[(X - mean)^3] / std^3
    let mean = client.mean(a, dims, true)?;
    let centered = client.sub(a, &mean)?;

    // Compute third moment: mean((centered)^3)
    let centered_cubed = client.pow_scalar(&centered, 3.0)?;
    let m3 = client.mean(&centered_cubed, dims, keepdim)?;

    // Compute std^3
    let std_val = client.std(a, dims, keepdim, correction)?;
    let std_cubed = client.pow_scalar(&std_val, 3.0)?;

    // skew = m3 / std^3 (with epsilon to avoid division by zero)
    let epsilon = Tensor::<CudaRuntime>::full_scalar(
        std_cubed.shape(),
        dtype,
        DIVISION_EPSILON,
        &client.device,
    );
    let std_cubed_safe = client.add(&std_cubed, &epsilon)?;

    client.div(&m3, &std_cubed_safe)
}

/// Compute kurtosis (fourth standardized moment, excess) using composition.
///
/// # Arguments
///
/// * `client` - The CUDA runtime client
/// * `a` - Input tensor
/// * `dims` - Dimensions to reduce (empty = all)
/// * `keepdim` - Whether to keep reduced dimensions
/// * `correction` - Bias correction (reserved for future use)
///
/// # Returns
///
/// Tensor containing the excess kurtosis values.
pub fn kurtosis_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();

    // kurtosis = E[(X - mean)^4] / std^4 - 3
    let mean = client.mean(a, dims, true)?;
    let centered = client.sub(a, &mean)?;

    // Compute fourth moment: mean((centered)^4)
    let centered_fourth = client.pow_scalar(&centered, 4.0)?;
    let m4 = client.mean(&centered_fourth, dims, keepdim)?;

    // Compute std^4
    let std_val = client.std(a, dims, keepdim, correction)?;
    let std_fourth = client.pow_scalar(&std_val, 4.0)?;

    // kurtosis = m4 / std^4 - 3 (with epsilon to avoid division by zero)
    let epsilon = Tensor::<CudaRuntime>::full_scalar(
        std_fourth.shape(),
        dtype,
        DIVISION_EPSILON,
        &client.device,
    );
    let std_fourth_safe = client.add(&std_fourth, &epsilon)?;

    let ratio = client.div(&m4, &std_fourth_safe)?;
    let three = Tensor::<CudaRuntime>::full_scalar(ratio.shape(), dtype, 3.0, &client.device);
    client.sub(&ratio, &three)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create bin edges tensor from computed f64 edges.
fn create_bin_edges(
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
fn tensor_to_f64(t: &Tensor<CudaRuntime>) -> Result<f64> {
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
