//! Top-level reduce dispatch: SIMD fast paths, narrow-float and integer widening.

use super::acc_kernel::reduce_kernel_acc;
use super::int_acc::{reduce_mean_int_kernel, reduce_sum_prod_int_kernel};
use super::scalar::reduce_kernel_scalar;
use crate::dtype::Element;
use crate::ops::{AccumulationPrecision, ReduceOp, max_identity, min_identity};

/// Reduce along contiguous dimension with automatic SIMD dispatch
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration
/// - AVX2: 8 f32s or 4 f64s per iteration
/// - Scalar fallback for other types or non-x86 platforms
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
    // A zero-length reduce dimension leaves `Max` and `Min` with no element to
    // seed from, and every path below seeds them by reading `a[o * 0]`. The
    // input allocation is empty here — `CpuRuntime::allocate` hands back a
    // dangling, non-null address for zero bytes — so that read is a silent
    // out-of-bounds access, not merely a wrong answer. `Sum`, `Mean`, `Prod`, `All` and `Any` all start from their
    // identity and iterate zero times, so they need no special case; these two
    // are given the identity of their own reduction instead.
    if reduce_size == 0 {
        match op {
            ReduceOp::Max => {
                // Floats fold to -inf, integers to the dtype's own minimum.
                let identity = T::from_f64(max_identity(T::DTYPE));
                for o in 0..outer_size {
                    *out.add(o) = identity;
                }
                return;
            }
            ReduceOp::Min => {
                // Floats fold to +inf, integers to the dtype's own maximum.
                let identity = T::from_f64(min_identity(T::DTYPE));
                for o in 0..outer_size {
                    *out.add(o) = identity;
                }
                return;
            }
            _ => {}
        }
    }

    // Dispatch to SIMD for f32/f64 on x86-64 and aarch64
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::reduce;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                reduce::reduce_f32(
                    op,
                    a as *const f32,
                    out as *mut f32,
                    reduce_size,
                    outer_size,
                );
                return;
            }
            DType::F64 => {
                reduce::reduce_f64(
                    op,
                    a as *const f64,
                    out as *mut f64,
                    reduce_size,
                    outer_size,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                reduce::reduce_f16(
                    op,
                    a as *const half::f16,
                    out as *mut half::f16,
                    reduce_size,
                    outer_size,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                reduce::reduce_bf16(
                    op,
                    a as *const half::bf16,
                    out as *mut half::bf16,
                    reduce_size,
                    outer_size,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback. A float narrower than F32 must never accumulate in its
    // own dtype: the running sum saturates and returns a constant. Widen to
    // f32 and narrow only the final result. This is what reaches FP8 on every
    // architecture, and F16/BF16 on architectures without the SIMD paths above.
    //
    // All/Any hold no accumulator, and routing them here would bounce back
    // into this function through `reduce_kernel_acc`.
    if T::DTYPE.is_narrow_float() && !matches!(op, ReduceOp::All | ReduceOp::Any) {
        reduce_kernel_acc::<T, f32>(op, a, out, reduce_size, outer_size);
        return;
    }

    // Integer `sum`, `prod`, and `mean` all build a running total wider than
    // one element, so accumulating in the element type wraps (release) or
    // panics (debug) on a total the output dtype cannot represent even though
    // the final result fits. Widen to i128, then narrow once with saturation
    // (`WideAcc`) — the same convention `cumsum`, `cumprod`, and `matmul` use.
    // `mean` additionally divides the wide sum before narrowing.
    if T::DTYPE.is_int() && matches!(op, ReduceOp::Sum | ReduceOp::Prod | ReduceOp::Mean) {
        match op {
            ReduceOp::Mean => reduce_mean_int_kernel(a, out, reduce_size, outer_size),
            _ => reduce_sum_prod_int_kernel(op, a, out, reduce_size, outer_size),
        }
        return;
    }

    reduce_kernel_scalar(op, a, out, reduce_size, outer_size);
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
            reduce_kernel_acc::<T, f32>(op, a, out, reduce_size, outer_size);
        }
        AccumulationPrecision::FP64 => {
            // Accumulate in f64 for maximum precision (math/science)
            reduce_kernel_acc::<T, f64>(op, a, out, reduce_size, outer_size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_sum() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 2];

        unsafe {
            // Reduce 3 elements per output, 2 outputs
            reduce_kernel(ReduceOp::Sum, a.as_ptr(), out.as_mut_ptr(), 3, 2);
        }

        assert_eq!(out, [6.0, 15.0]); // [1+2+3, 4+5+6]
    }

    #[test]
    fn test_reduce_mean() {
        let a = [1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        let mut out = [0.0f32; 2];

        unsafe {
            reduce_kernel(ReduceOp::Mean, a.as_ptr(), out.as_mut_ptr(), 3, 2);
        }

        assert_eq!(out, [2.0, 20.0]); // [6/3, 60/3]
    }

    #[test]
    fn test_reduce_max() {
        let a = [1.0f32, 5.0, 3.0, 2.0, 8.0, 4.0];
        let mut out = [0.0f32; 2];

        unsafe {
            reduce_kernel(ReduceOp::Max, a.as_ptr(), out.as_mut_ptr(), 3, 2);
        }

        assert_eq!(out, [5.0, 8.0]);
    }
}
