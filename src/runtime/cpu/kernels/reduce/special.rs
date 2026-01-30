//! Special reduction operations
//!
//! Contains argmax, argmin, softmax, and variance kernels.

use crate::dtype::Element;

/// Argmax along a dimension - returns indices of maximum values
///
/// # Arguments
/// * `a` - Input pointer (outer_size * reduce_size * inner_size elements)
/// * `out` - Output pointer (outer_size * inner_size elements) for i64 indices
/// * `outer_size` - Product of dimensions before the reduction dimension
/// * `reduce_size` - Size of the dimension being reduced
/// * `inner_size` - Product of dimensions after the reduction dimension
///
/// # Safety
/// - `a` must point to `outer_size * reduce_size * inner_size` elements
/// - `out` must point to `outer_size * inner_size` i64 elements
#[inline]
pub unsafe fn argmax_kernel<T: Element>(
    a: *const T,
    out: *mut i64,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Base index for this (outer, inner) position
            let base_idx = outer * reduce_size * inner_size + inner;

            // Find index of maximum value
            let mut max_val = (*a.add(base_idx)).to_f64();
            let mut max_idx: i64 = 0;

            for r in 1..reduce_size {
                let idx = base_idx + r * inner_size;
                let val = (*a.add(idx)).to_f64();
                if val > max_val {
                    max_val = val;
                    max_idx = r as i64;
                }
            }

            *out.add(outer * inner_size + inner) = max_idx;
        }
    }
}

/// Argmin along a dimension - returns indices of minimum values
///
/// # Arguments
/// * `a` - Input pointer (outer_size * reduce_size * inner_size elements)
/// * `out` - Output pointer (outer_size * inner_size elements) for i64 indices
/// * `outer_size` - Product of dimensions before the reduction dimension
/// * `reduce_size` - Size of the dimension being reduced
/// * `inner_size` - Product of dimensions after the reduction dimension
///
/// # Safety
/// - `a` must point to `outer_size * reduce_size * inner_size` elements
/// - `out` must point to `outer_size * inner_size` i64 elements
#[inline]
pub unsafe fn argmin_kernel<T: Element>(
    a: *const T,
    out: *mut i64,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) {
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Base index for this (outer, inner) position
            let base_idx = outer * reduce_size * inner_size + inner;

            // Find index of minimum value
            let mut min_val = (*a.add(base_idx)).to_f64();
            let mut min_idx: i64 = 0;

            for r in 1..reduce_size {
                let idx = base_idx + r * inner_size;
                let val = (*a.add(idx)).to_f64();
                if val < min_val {
                    min_val = val;
                    min_idx = r as i64;
                }
            }

            *out.add(outer * inner_size + inner) = min_idx;
        }
    }
}

/// Softmax along the last dimension
///
/// softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
///
/// # Arguments
/// * `a` - Input pointer (outer_size * dim_size elements)
/// * `out` - Output pointer (outer_size * dim_size elements)
/// * `outer_size` - Number of independent softmax operations
/// * `dim_size` - Size of the softmax dimension
///
/// # Safety
/// - `a` and `out` must point to `outer_size * dim_size` elements
#[inline]
pub unsafe fn softmax_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    outer_size: usize,
    dim_size: usize,
) {
    for o in 0..outer_size {
        let base = o * dim_size;

        // Find max for numerical stability
        let mut max_val = (*a.add(base)).to_f64();
        for d in 1..dim_size {
            let val = (*a.add(base + d)).to_f64();
            if val > max_val {
                max_val = val;
            }
        }

        // Compute exp(x - max) and sum
        let mut sum = 0.0f64;
        for d in 0..dim_size {
            let val = (*a.add(base + d)).to_f64();
            let exp_val = (val - max_val).exp();
            *out.add(base + d) = T::from_f64(exp_val);
            sum += exp_val;
        }

        // Normalize by sum
        let inv_sum = 1.0 / sum;
        for d in 0..dim_size {
            let val = (*out.add(base + d)).to_f64();
            *out.add(base + d) = T::from_f64(val * inv_sum);
        }
    }
}

/// Compute variance along a dimension
///
/// variance = sum((x - mean)^2) / (N - correction)
///
/// # Arguments
/// * `a` - Input data pointer
/// * `out` - Output pointer (for variance values)
/// * `outer_size` - Product of dimensions before the reduce dimension
/// * `reduce_size` - Size of the dimension being reduced
/// * `inner_size` - Product of dimensions after the reduce dimension
/// * `correction` - Degrees of freedom correction (0 for population, 1 for sample)
///
/// # Safety
/// - `a` must point to `outer_size * reduce_size * inner_size` elements
/// - `out` must point to `outer_size * inner_size` elements
#[inline]
pub unsafe fn variance_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
    correction: usize,
) {
    let total_size = reduce_size;
    let divisor = (total_size.saturating_sub(correction)).max(1) as f64;

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // First pass: compute mean
            let mut sum = 0.0f64;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                sum += (*a.add(idx)).to_f64();
            }
            let mean = sum / (reduce_size as f64);

            // Second pass: compute variance
            let mut var_sum = 0.0f64;
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                let diff = (*a.add(idx)).to_f64() - mean;
                var_sum += diff * diff;
            }

            let out_idx = outer * inner_size + inner;
            *out.add(out_idx) = T::from_f64(var_sum / divisor);
        }
    }
}
