//! Contiguous and strided logsumexp dispatch with original-value masked limits.

use crate::dtype::Element;

/// Log-sum-exp along a contiguous dimension (numerically stable)
///
/// Computes log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration with vectorized exp
/// - AVX2: 8 f32s or 4 f64s per iteration with vectorized exp
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Arguments
/// * `a` - Input pointer (reduce_size * outer_size elements, contiguous)
/// * `out` - Output pointer (outer_size elements)
/// * `reduce_size` - Number of elements to reduce per segment
/// * `outer_size` - Number of independent reductions
///
/// # Safety
/// - `a` must point to `reduce_size * outer_size` elements
/// - `out` must point to `outer_size` elements
#[inline]
pub unsafe fn logsumexp_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    // A zero-length reduce dimension makes every output element a fold over no
    // input: `log(sum of nothing)` is `log(0)`, so the answer is -inf. Both the
    // SIMD paths below and the scalar fallback seed their max from `a[base]`,
    // which on an empty allocation reads past the end (`CpuRuntime::allocate`
    // hands back a dangling, non-null address for zero bytes), so this must
    // answer before any of them run.
    if reduce_size == 0 {
        let identity = T::from_f64(f64::NEG_INFINITY);
        for o in 0..outer_size {
            *out.add(o) = identity;
        }
        return;
    }

    // Dispatch to SIMD for f32/f64 on x86-64 and aarch64
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::logsumexp;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                logsumexp::logsumexp_f32(a as *const f32, out as *mut f32, reduce_size, outer_size);
                return;
            }
            DType::F64 => {
                logsumexp::logsumexp_f64(a as *const f64, out as *mut f64, reduce_size, outer_size);
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                logsumexp::logsumexp_f16(
                    a as *const half::f16,
                    out as *mut half::f16,
                    reduce_size,
                    outer_size,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                logsumexp::logsumexp_bf16(
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

    // Scalar fallback
    logsumexp_kernel_scalar(a, out, reduce_size, outer_size);
}

/// Scalar logsumexp for all Element types
#[inline]
unsafe fn logsumexp_kernel_scalar<T: Element>(
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    for o in 0..outer_size {
        let base = o * reduce_size;

        // Step 1: Find max
        let mut max_val = *a.add(base);
        let mut all_negative_infinity = max_val.to_f64() == f64::NEG_INFINITY;
        for i in 1..reduce_size {
            let val = *a.add(base + i);
            all_negative_infinity &= val.to_f64() == f64::NEG_INFINITY;
            if val > max_val {
                max_val = val;
            }
        }

        if all_negative_infinity {
            *out.add(o) = T::from_f64(f64::NEG_INFINITY);
            continue;
        }

        // Step 2: Compute sum(exp(x - max))
        //
        // A float narrower than F32 must not hold this accumulator. Every term
        // is at most 1.0, so the running sum saturates quickly and stalls, and
        // `log(sum)` then comes back short — which makes every log_softmax
        // built on it wrong. Widen those to f64 (what `logsumexp_strided_kernel`
        // already does) and narrow only the final result. Every other dtype
        // keeps its own accumulator and its existing result.
        if T::DTYPE.is_narrow_float() {
            let mut sum = 0.0f64;
            for i in 0..reduce_size {
                let val = (*a.add(base + i)).to_f64();
                sum += (val - max_val.to_f64()).exp();
            }
            *out.add(o) = T::from_f64(max_val.to_f64() + sum.ln());
            continue;
        }

        let mut sum = T::zero();
        for i in 0..reduce_size {
            let val = *a.add(base + i);
            // Compute exp(val - max_val) using f64 for precision
            let exp_val = T::from_f64((val.to_f64() - max_val.to_f64()).exp());
            sum = sum + exp_val;
        }

        // Step 3: Result = max + log(sum)
        *out.add(o) = T::from_f64(max_val.to_f64() + sum.to_f64().ln());
    }
}

/// Log-sum-exp along a strided dimension (numerically stable)
///
/// # Safety
/// - Pointers must be valid for the given strides and sizes
#[inline]
pub unsafe fn logsumexp_strided_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
    inner_size: usize,
    _in_stride: usize, // stride along the reduce dimension in input (unused, kept for API parity)
    out_stride: usize, // stride in output
) {
    // Same identity as `logsumexp_kernel`: a fold over no input is `log(0)`,
    // and the seed read below would otherwise walk off an empty allocation.
    if reduce_size == 0 {
        let identity = T::from_f64(f64::NEG_INFINITY);
        for o in 0..outer_size {
            for i in 0..inner_size {
                *out.add(o * out_stride + i) = identity;
            }
        }
        return;
    }

    for o in 0..outer_size {
        for i in 0..inner_size {
            let out_idx = o * out_stride + i;

            // Step 1: Find max along reduce dimension
            let first_idx = o * reduce_size * inner_size + i;
            let mut max_val = *a.add(first_idx);
            let mut all_negative_infinity = max_val.to_f64() == f64::NEG_INFINITY;
            for r in 1..reduce_size {
                let idx = o * reduce_size * inner_size + r * inner_size + i;
                let val = *a.add(idx);
                all_negative_infinity &= val.to_f64() == f64::NEG_INFINITY;
                if val > max_val {
                    max_val = val;
                }
            }

            if all_negative_infinity {
                *out.add(out_idx) = T::from_f64(f64::NEG_INFINITY);
                continue;
            }

            // Step 2: Compute sum(exp(x - max))
            let mut sum = 0.0f64;
            for r in 0..reduce_size {
                let idx = o * reduce_size * inner_size + r * inner_size + i;
                let val = (*a.add(idx)).to_f64();
                sum += (val - max_val.to_f64()).exp();
            }

            // Step 3: Result = max + log(sum)
            *out.add(out_idx) = T::from_f64(max_val.to_f64() + sum.ln());
        }
    }
}

#[cfg(test)]
mod tests;
