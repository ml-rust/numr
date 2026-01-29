//! Statistics operations for CPU runtime
//!
//! Implements quantile, percentile, median, histogram, skewness, and kurtosis.
//!
//! This module uses optimized CPU kernels with direct memory access for maximum
//! performance, while sharing common logic with other backends via the
//! `statistics_common` module.

use super::helpers::{dispatch_dtype, ensure_contiguous};
use super::sort::sort_impl;
use super::{CpuClient, CpuRuntime};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::ops::{ScalarOps, TensorOps, compute_reduce_strides, reduce_dim_output_shape};
use crate::runtime::normalize_dim;
use crate::runtime::statistics_common::{
    self, DIVISION_EPSILON, Interpolation, compute_bin_edges_f64, compute_kurtosis,
    compute_skewness,
};
use crate::tensor::Tensor;

// ============================================================================
// Public API Functions
// ============================================================================

/// Compute quantile along a dimension.
///
/// # Arguments
///
/// * `client` - The CPU runtime client
/// * `a` - Input tensor
/// * `q` - Quantile to compute, must be in [0.0, 1.0]
/// * `dim` - Dimension to reduce along (None = flatten first)
/// * `keepdim` - Whether to keep the reduced dimension
/// * `interpolation` - Interpolation method ("linear", "lower", "higher", "nearest", "midpoint")
///
/// # Returns
///
/// Tensor containing the quantile values.
///
/// # Errors
///
/// - `InvalidArgument` if q is not in [0, 1]
/// - `InvalidArgument` if interpolation method is invalid
/// - `InvalidDimension` if dim is out of bounds
pub fn quantile_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    q: f64,
    dim: Option<isize>,
    keepdim: bool,
    interpolation: &str,
) -> Result<Tensor<CpuRuntime>> {
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
            return Ok(Tensor::<CpuRuntime>::empty(
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
        return Ok(Tensor::<CpuRuntime>::empty(
            &out_shape,
            dtype,
            &client.device,
        ));
    }

    // Sort along the dimension
    let sorted = sort_impl(client, a, dim_val, false)?;

    // Compute output shape and strides
    let out_shape = reduce_dim_output_shape(shape, dim_idx, keepdim);
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);
    let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim_idx);

    let sorted_contig = ensure_contiguous(&sorted);
    let sorted_ptr = sorted_contig.storage().ptr();
    let out_ptr = out.storage().ptr();

    // Dispatch to typed kernel
    dispatch_dtype!(dtype, T => {
        unsafe {
            quantile_kernel::<T>(
                sorted_ptr as *const T,
                out_ptr as *mut T,
                outer_size,
                reduce_size,
                inner_size,
                q,
                interp,
            );
        }
    }, "quantile");

    Ok(out)
}

/// Compute percentile (quantile * 100) along a dimension.
///
/// # Arguments
///
/// * `client` - The CPU runtime client
/// * `a` - Input tensor
/// * `p` - Percentile to compute, must be in [0.0, 100.0]
/// * `dim` - Dimension to reduce along (None = flatten first)
/// * `keepdim` - Whether to keep the reduced dimension
///
/// # Returns
///
/// Tensor containing the percentile values.
pub fn percentile_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    p: f64,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<Tensor<CpuRuntime>> {
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
/// * `client` - The CPU runtime client
/// * `a` - Input tensor
/// * `dim` - Dimension to reduce along (None = flatten first)
/// * `keepdim` - Whether to keep the reduced dimension
///
/// # Returns
///
/// Tensor containing the median values.
pub fn median_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<Tensor<CpuRuntime>> {
    quantile_impl(client, a, 0.5, dim, keepdim, "linear")
}

/// Compute histogram of values.
///
/// # Arguments
///
/// * `client` - The CPU runtime client
/// * `a` - Input tensor (will be flattened)
/// * `bins` - Number of histogram bins (must be > 0)
/// * `range` - Optional (min, max) range; defaults to (a.min(), a.max())
///
/// # Returns
///
/// Tuple of (histogram counts as I64 tensor, bin edges tensor).
///
/// # Errors
///
/// - `InvalidArgument` if bins is 0
/// - `InvalidArgument` if range min >= max
pub fn histogram_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    bins: usize,
    range: Option<(f64, f64)>,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
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
        let hist = Tensor::<CpuRuntime>::zeros(&[bins], DType::I64, &client.device);
        let edges = create_bin_edges(client, min_val, max_val, bins, dtype)?;
        return Ok((hist, edges));
    }

    // Flatten input
    let flat = a.reshape(&[numel])?;
    let flat_contig = ensure_contiguous(&flat);
    let flat_ptr = flat_contig.storage().ptr();

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

    // Create histogram counts tensor
    let hist = Tensor::<CpuRuntime>::zeros(&[bins], DType::I64, &client.device);
    let hist_ptr = hist.storage().ptr() as *mut i64;

    // Compute histogram using optimized kernel
    dispatch_dtype!(dtype, T => {
        unsafe {
            histogram_kernel::<T>(flat_ptr as *const T, hist_ptr, numel, bins, min_val, max_val);
        }
    }, "histogram");

    // Create bin edges
    let edges = create_bin_edges(client, min_val, max_val, bins, dtype)?;

    Ok((hist, edges))
}

/// Compute skewness (third standardized moment) along dimensions.
///
/// Skewness measures the asymmetry of a distribution.
///
/// # Formula
///
/// ```text
/// skewness = E[(X - μ)³] / σ³
/// ```
///
/// # Arguments
///
/// * `client` - The CPU runtime client
/// * `a` - Input tensor
/// * `dims` - Dimensions to reduce (empty = all)
/// * `keepdim` - Whether to keep reduced dimensions
/// * `correction` - Bias correction (reserved for future use)
///
/// # Returns
///
/// Tensor containing the skewness values.
pub fn skew_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Handle scalar/global reduction case
    if dims.is_empty() {
        let numel = a.numel();
        let a_contig = ensure_contiguous(a);
        let a_ptr = a_contig.storage().ptr();

        let skewness = dispatch_dtype!(dtype, T => {
            unsafe {
                let slice = std::slice::from_raw_parts(a_ptr as *const T, numel);
                compute_skewness(slice, correction)
            }
        }, "skew");

        let out_shape = if keepdim { vec![1; ndim] } else { vec![] };
        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe { *(out_ptr as *mut T) = T::from_f64(skewness); }
        }, "skew");

        return Ok(out);
    }

    // Use composition for dimensional reduction: mean, centered, std, pow, div
    let mean = client.mean(a, dims, true)?;
    let centered = client.sub(a, &mean)?;

    // Compute third moment: mean((centered)^3)
    let centered_cubed = client.pow_scalar(&centered, 3.0)?;
    let m3 = client.mean(&centered_cubed, dims, keepdim)?;

    // Compute std^3
    let std_val = client.std(a, dims, keepdim, correction)?;
    let std_cubed = client.pow_scalar(&std_val, 3.0)?;

    // skew = m3 / std^3 (with epsilon to avoid division by zero)
    let epsilon = Tensor::<CpuRuntime>::full_scalar(
        std_cubed.shape(),
        dtype,
        DIVISION_EPSILON,
        &client.device,
    );
    let std_cubed_safe = client.add(&std_cubed, &epsilon)?;

    client.div(&m3, &std_cubed_safe)
}

/// Compute kurtosis (fourth standardized moment, excess) along dimensions.
///
/// Excess kurtosis measures the "tailedness" of a distribution relative to
/// a normal distribution.
///
/// # Formula
///
/// ```text
/// excess_kurtosis = E[(X - μ)⁴] / σ⁴ - 3
/// ```
///
/// # Arguments
///
/// * `client` - The CPU runtime client
/// * `a` - Input tensor
/// * `dims` - Dimensions to reduce (empty = all)
/// * `keepdim` - Whether to keep reduced dimensions
/// * `correction` - Bias correction (reserved for future use)
///
/// # Returns
///
/// Tensor containing the excess kurtosis values.
pub fn kurtosis_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Handle scalar/global reduction case
    if dims.is_empty() {
        let numel = a.numel();
        let a_contig = ensure_contiguous(a);
        let a_ptr = a_contig.storage().ptr();

        let kurtosis = dispatch_dtype!(dtype, T => {
            unsafe {
                let slice = std::slice::from_raw_parts(a_ptr as *const T, numel);
                compute_kurtosis(slice, correction)
            }
        }, "kurtosis");

        let out_shape = if keepdim { vec![1; ndim] } else { vec![] };
        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);
        let out_ptr = out.storage().ptr();

        dispatch_dtype!(dtype, T => {
            unsafe { *(out_ptr as *mut T) = T::from_f64(kurtosis); }
        }, "kurtosis");

        return Ok(out);
    }

    // Use composition for dimensional reduction
    let mean = client.mean(a, dims, true)?;
    let centered = client.sub(a, &mean)?;

    // Compute fourth moment: mean((centered)^4)
    let centered_fourth = client.pow_scalar(&centered, 4.0)?;
    let m4 = client.mean(&centered_fourth, dims, keepdim)?;

    // Compute std^4
    let std_val = client.std(a, dims, keepdim, correction)?;
    let std_fourth = client.pow_scalar(&std_val, 4.0)?;

    // kurtosis = m4 / std^4 - 3 (with epsilon to avoid division by zero)
    let epsilon = Tensor::<CpuRuntime>::full_scalar(
        std_fourth.shape(),
        dtype,
        DIVISION_EPSILON,
        &client.device,
    );
    let std_fourth_safe = client.add(&std_fourth, &epsilon)?;

    let ratio = client.div(&m4, &std_fourth_safe)?;
    let three = Tensor::<CpuRuntime>::full_scalar(ratio.shape(), dtype, 3.0, &client.device);
    client.sub(&ratio, &three)
}

// Note: cov and corrcoef are implemented via LinearAlgebraAlgorithms trait,
// not as separate functions here. The TensorOps methods delegate directly
// to the trait implementations.

// ============================================================================
// Optimized CPU Kernels
// ============================================================================

/// Quantile kernel - computes quantile from pre-sorted data.
///
/// # Safety
///
/// Caller must ensure:
/// - `sorted` points to valid memory of size `outer_size * reduce_size * inner_size`
/// - `out` points to valid memory of size `outer_size * inner_size`
/// - `reduce_size > 0`
/// - `q` is in [0.0, 1.0]
#[inline]
unsafe fn quantile_kernel<T: Element>(
    sorted: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
    q: f64,
    interp: Interpolation,
) {
    if reduce_size == 0 {
        return;
    }

    // Validate bounds in debug builds
    let total_input_size = outer_size * reduce_size * inner_size;
    let total_output_size = outer_size * inner_size;
    debug_assert!(
        total_input_size <= isize::MAX as usize,
        "Input array too large"
    );
    debug_assert!(
        total_output_size <= isize::MAX as usize,
        "Output array too large"
    );

    // Compute quantile indices using shared logic
    let (floor_idx, ceil_idx, frac) = statistics_common::compute_quantile_indices(q, reduce_size);

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let base_idx = outer * reduce_size * inner_size + inner;
            let out_idx = outer * inner_size + inner;

            // Debug bounds checks
            debug_assert!(
                base_idx + ceil_idx * inner_size < total_input_size,
                "Input index {} out of bounds (size {})",
                base_idx + ceil_idx * inner_size,
                total_input_size
            );
            debug_assert!(
                out_idx < total_output_size,
                "Output index {} out of bounds (size {})",
                out_idx,
                total_output_size
            );

            // SAFETY: bounds checked by debug_assert! above
            let lower_val = unsafe { (*sorted.add(base_idx + floor_idx * inner_size)).to_f64() };
            let upper_val = unsafe { (*sorted.add(base_idx + ceil_idx * inner_size)).to_f64() };
            let value = interp.interpolate(lower_val, upper_val, frac);

            // SAFETY: out_idx bounds checked by debug_assert! above
            unsafe { *out.add(out_idx) = T::from_f64(value) };
        }
    }
}

/// Histogram kernel - counts values into bins.
///
/// # Safety
///
/// Caller must ensure:
/// - `data` points to valid memory of size `numel`
/// - `counts` points to zero-initialized memory of size `bins`
/// - `bins > 0`
/// - `max_val > min_val`
#[inline]
unsafe fn histogram_kernel<T: Element>(
    data: *const T,
    counts: *mut i64,
    numel: usize,
    bins: usize,
    min_val: f64,
    max_val: f64,
) {
    debug_assert!(bins > 0, "bins must be positive");
    debug_assert!(max_val > min_val, "max_val must be greater than min_val");

    let bin_width = (max_val - min_val) / bins as f64;

    for i in 0..numel {
        // SAFETY: i < numel, and caller guarantees data has numel elements
        let val = unsafe { (*data.add(i)).to_f64() };
        let bin_idx = statistics_common::compute_bin_index(val, min_val, bin_width, bins);

        debug_assert!(
            bin_idx < bins,
            "bin_idx {} out of bounds (bins {})",
            bin_idx,
            bins
        );

        // SAFETY: bin_idx < bins is guaranteed by compute_bin_index clamping
        unsafe { *counts.add(bin_idx) += 1 };
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create bin edges tensor from computed f64 edges.
fn create_bin_edges(
    client: &CpuClient,
    min_val: f64,
    max_val: f64,
    bins: usize,
    dtype: DType,
) -> Result<Tensor<CpuRuntime>> {
    let edges_data = compute_bin_edges_f64(min_val, max_val, bins);

    // Create tensor and copy data based on dtype
    let edges = Tensor::<CpuRuntime>::empty(&[bins + 1], dtype, &client.device);
    let edges_ptr = edges.storage().ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            let out_slice = std::slice::from_raw_parts_mut(edges_ptr as *mut T, bins + 1);
            for (i, &val) in edges_data.iter().enumerate() {
                out_slice[i] = T::from_f64(val);
            }
        }
    }, "histogram_edges");

    Ok(edges)
}

/// Extract scalar f64 value from tensor.
fn tensor_to_f64(t: &Tensor<CpuRuntime>) -> Result<f64> {
    let dtype = t.dtype();
    let ptr = t.storage().ptr();

    let val = dispatch_dtype!(dtype, T => {
        unsafe { (*(ptr as *const T)).to_f64() }
    }, "tensor_to_f64");

    Ok(val)
}
