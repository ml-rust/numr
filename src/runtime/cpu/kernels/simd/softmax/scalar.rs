//! Scalar fallback implementations of the online softmax algorithm.

/// Scalar softmax for f32 using online algorithm (2-pass).
#[inline]
pub(super) unsafe fn softmax_scalar_f32(
    a: *const f32,
    out: *mut f32,
    outer_size: usize,
    dim_size: usize,
) {
    for o in 0..outer_size {
        let base = o * dim_size;

        // Pass 1: Online max + sum — single read of input
        let mut max_val = *a.add(base);
        let mut sum = if max_val.is_finite() { 1.0f32 } else { 0.0f32 };
        for d in 1..dim_size {
            let val = *a.add(base + d);
            if val > max_val {
                // Guard: if max_val == -inf, rescale factor is 0 (not NaN)
                let rescale = if max_val == f32::NEG_INFINITY {
                    0.0
                } else {
                    (max_val - val).exp()
                };
                sum = sum * rescale + 1.0;
                max_val = val;
            } else if val == f32::NEG_INFINITY {
                // exp(-inf - anything) = 0, skip to avoid NaN from -inf - (-inf)
            } else {
                sum += (val - max_val).exp();
            }
        }

        // Pass 2: Compute exp(x - max) / sum — one read of input, one write of output
        let inv_sum = 1.0 / sum;
        for d in 0..dim_size {
            let val = *a.add(base + d);
            *out.add(base + d) = if val == f32::NEG_INFINITY {
                0.0
            } else {
                (val - max_val).exp() * inv_sum
            };
        }
    }
}

/// Scalar softmax for f64 using online algorithm (2-pass).
#[inline]
pub(super) unsafe fn softmax_scalar_f64(
    a: *const f64,
    out: *mut f64,
    outer_size: usize,
    dim_size: usize,
) {
    for o in 0..outer_size {
        let base = o * dim_size;

        // Pass 1: Online max + sum
        let mut max_val = *a.add(base);
        let mut sum = if max_val.is_finite() { 1.0f64 } else { 0.0f64 };
        for d in 1..dim_size {
            let val = *a.add(base + d);
            if val > max_val {
                let rescale = if max_val == f64::NEG_INFINITY {
                    0.0
                } else {
                    (max_val - val).exp()
                };
                sum = sum * rescale + 1.0;
                max_val = val;
            } else if val == f64::NEG_INFINITY {
                // exp(-inf - anything) = 0, skip to avoid NaN from -inf - (-inf)
            } else {
                sum += (val - max_val).exp();
            }
        }

        // Pass 2: Compute exp(x - max) / sum
        let inv_sum = 1.0 / sum;
        for d in 0..dim_size {
            let val = *a.add(base + d);
            *out.add(base + d) = if val == f64::NEG_INFINITY {
                0.0
            } else {
                (val - max_val).exp() * inv_sum
            };
        }
    }
}
