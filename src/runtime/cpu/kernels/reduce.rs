//! Reduction operation kernels

use crate::dtype::Element;
use crate::ops::{AccumulationPrecision, ReduceOp};

/// Reduce along contiguous dimension
///
/// # Arguments
/// * `op` - Reduction operation
/// * `a` - Input pointer (reduce_size * outer_size elements)
/// * `out` - Output pointer (outer_size elements)
/// * `reduce_size` - Number of elements to reduce over
/// * `outer_size` - Number of independent reductions
///
/// # Safety
/// - `a` must point to `reduce_size * outer_size` elements
/// - `out` must point to `outer_size` elements
#[inline]
pub unsafe fn reduce_kernel<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    match op {
        ReduceOp::Sum => {
            for o in 0..outer_size {
                let mut sum = T::zero();
                for r in 0..reduce_size {
                    sum = sum + *a.add(o * reduce_size + r);
                }
                *out.add(o) = sum;
            }
        }
        ReduceOp::Mean => {
            let scale = 1.0 / reduce_size as f64;
            for o in 0..outer_size {
                let mut sum = T::zero();
                for r in 0..reduce_size {
                    sum = sum + *a.add(o * reduce_size + r);
                }
                *out.add(o) = T::from_f64(sum.to_f64() * scale);
            }
        }
        ReduceOp::Max => {
            for o in 0..outer_size {
                let mut max_val = *a.add(o * reduce_size);
                for r in 1..reduce_size {
                    let val = *a.add(o * reduce_size + r);
                    if val > max_val {
                        max_val = val;
                    }
                }
                *out.add(o) = max_val;
            }
        }
        ReduceOp::Min => {
            for o in 0..outer_size {
                let mut min_val = *a.add(o * reduce_size);
                for r in 1..reduce_size {
                    let val = *a.add(o * reduce_size + r);
                    if val < min_val {
                        min_val = val;
                    }
                }
                *out.add(o) = min_val;
            }
        }
        ReduceOp::Prod => {
            for o in 0..outer_size {
                let mut prod = T::one();
                for r in 0..reduce_size {
                    prod = prod * *a.add(o * reduce_size + r);
                }
                *out.add(o) = prod;
            }
        }
        ReduceOp::All | ReduceOp::Any => {
            // Boolean reductions - convert to/from f64 (0.0 = false, non-zero = true)
            let is_any = matches!(op, ReduceOp::Any);
            for o in 0..outer_size {
                let mut result = !is_any; // All starts true, Any starts false
                for r in 0..reduce_size {
                    let val = (*a.add(o * reduce_size + r)).to_f64() != 0.0;
                    if is_any {
                        result = result || val;
                    } else {
                        result = result && val;
                    }
                }
                *out.add(o) = T::from_f64(if result { 1.0 } else { 0.0 });
            }
        }
    }
}

/// Reduce kernel with explicit accumulation precision
///
/// For reduced-precision types (F16, BF16, FP8), this allows accumulating
/// in a higher precision format for better numerical stability.
///
/// # Arguments
/// * `op` - Reduction operation
/// * `a` - Input pointer (reduce_size * outer_size elements)
/// * `out` - Output pointer (outer_size elements)
/// * `reduce_size` - Number of elements to reduce over
/// * `outer_size` - Number of independent reductions
/// * `precision` - Accumulation precision
///
/// # Safety
/// - `a` must point to `reduce_size * outer_size` elements
/// - `out` must point to `outer_size` elements
#[inline]
pub unsafe fn reduce_kernel_with_precision<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
    precision: AccumulationPrecision,
) {
    match precision {
        AccumulationPrecision::Native => {
            // Use native type accumulation (existing behavior)
            reduce_kernel(op, a, out, reduce_size, outer_size);
        }
        AccumulationPrecision::FP32 | AccumulationPrecision::BF16 => {
            // Accumulate in f32 for better precision
            // BF16 uses f32 on CPU since there's no native bf16 arithmetic
            reduce_kernel_f32_acc(op, a, out, reduce_size, outer_size);
        }
        AccumulationPrecision::FP64 => {
            // Accumulate in f64 for maximum precision (math/science)
            reduce_kernel_f64_acc(op, a, out, reduce_size, outer_size);
        }
    }
}

/// Trait for accumulation types (f32, f64) used in precision-aware reductions.
///
/// This allows a single generic implementation for both FP32 and FP64 accumulation,
/// avoiding code duplication while maintaining type safety and performance.
///
/// Uses `Into<f64>` for output conversion, `acc_in` for input (f64 -> Self).
pub trait Accumulator: Copy + PartialOrd + PartialEq + Into<f64> {
    const ZERO: Self;
    const ONE: Self;
    /// Convert f64 input to accumulator type
    fn acc_in(v: f64) -> Self;
    fn acc_add(self, other: Self) -> Self;
    fn acc_mul(self, other: Self) -> Self;
    fn acc_div(self, n: usize) -> Self;
}

impl Accumulator for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn acc_in(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn acc_add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn acc_mul(self, other: Self) -> Self {
        self * other
    }
    #[inline]
    fn acc_div(self, n: usize) -> Self {
        self / n as f32
    }
}

impl Accumulator for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn acc_in(v: f64) -> Self {
        v
    }
    #[inline]
    fn acc_add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn acc_mul(self, other: Self) -> Self {
        self * other
    }
    #[inline]
    fn acc_div(self, n: usize) -> Self {
        self / n as f64
    }
}

/// Generic reduce kernel with configurable accumulation precision.
///
/// Converts input elements to accumulator type A, performs reduction, then converts back to T.
#[inline]
unsafe fn reduce_kernel_acc<T: Element, A: Accumulator>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    match op {
        ReduceOp::Sum => {
            for o in 0..outer_size {
                let mut sum = A::ZERO;
                for r in 0..reduce_size {
                    sum = sum.acc_add(A::acc_in((*a.add(o * reduce_size + r)).to_f64()));
                }
                *out.add(o) = T::from_f64(sum.into());
            }
        }
        ReduceOp::Mean => {
            for o in 0..outer_size {
                let mut sum = A::ZERO;
                for r in 0..reduce_size {
                    sum = sum.acc_add(A::acc_in((*a.add(o * reduce_size + r)).to_f64()));
                }
                *out.add(o) = T::from_f64(sum.acc_div(reduce_size).into());
            }
        }
        ReduceOp::Max => {
            for o in 0..outer_size {
                let mut max_val = A::acc_in((*a.add(o * reduce_size)).to_f64());
                for r in 1..reduce_size {
                    let val = A::acc_in((*a.add(o * reduce_size + r)).to_f64());
                    if val > max_val {
                        max_val = val;
                    }
                }
                *out.add(o) = T::from_f64(max_val.into());
            }
        }
        ReduceOp::Min => {
            for o in 0..outer_size {
                let mut min_val = A::acc_in((*a.add(o * reduce_size)).to_f64());
                for r in 1..reduce_size {
                    let val = A::acc_in((*a.add(o * reduce_size + r)).to_f64());
                    if val < min_val {
                        min_val = val;
                    }
                }
                *out.add(o) = T::from_f64(min_val.into());
            }
        }
        ReduceOp::Prod => {
            for o in 0..outer_size {
                let mut prod = A::ONE;
                for r in 0..reduce_size {
                    prod = prod.acc_mul(A::acc_in((*a.add(o * reduce_size + r)).to_f64()));
                }
                *out.add(o) = T::from_f64(prod.into());
            }
        }
        ReduceOp::All | ReduceOp::Any => {
            // Boolean reductions don't benefit from higher precision accumulation
            reduce_kernel(op, a, out, reduce_size, outer_size);
        }
    }
}

/// Reduce kernel with f32 accumulation (convenience wrapper)
#[inline]
unsafe fn reduce_kernel_f32_acc<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    reduce_kernel_acc::<T, f32>(op, a, out, reduce_size, outer_size)
}

/// Reduce kernel with f64 accumulation (convenience wrapper)
#[inline]
unsafe fn reduce_kernel_f64_acc<T: Element>(
    op: ReduceOp,
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    reduce_kernel_acc::<T, f64>(op, a, out, reduce_size, outer_size)
}

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
