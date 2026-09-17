//! Scalar fallbacks for logsumexp.

/// Scalar logsumexp for f32
#[inline]
pub(super) unsafe fn logsumexp_scalar_f32(
    a: *const f32,
    out: *mut f32,
    reduce_size: usize,
    outer_size: usize,
) {
    for o in 0..outer_size {
        let base = o * reduce_size;

        // Find max
        let mut max_val = *a.add(base);
        for i in 1..reduce_size {
            let val = *a.add(base + i);
            if val > max_val {
                max_val = val;
            }
        }

        // Compute sum(exp(x - max))
        let mut sum = 0.0f32;
        for i in 0..reduce_size {
            let val = *a.add(base + i);
            sum += (val - max_val).exp();
        }

        // Result = max + log(sum)
        *out.add(o) = max_val + sum.ln();
    }
}

/// Scalar logsumexp for f64
#[inline]
pub(super) unsafe fn logsumexp_scalar_f64(
    a: *const f64,
    out: *mut f64,
    reduce_size: usize,
    outer_size: usize,
) {
    for o in 0..outer_size {
        let base = o * reduce_size;

        // Find max
        let mut max_val = *a.add(base);
        for i in 1..reduce_size {
            let val = *a.add(base + i);
            if val > max_val {
                max_val = val;
            }
        }

        // Compute sum(exp(x - max))
        let mut sum = 0.0f64;
        for i in 0..reduce_size {
            let val = *a.add(base + i);
            sum += (val - max_val).exp();
        }

        // Result = max + log(sum)
        *out.add(o) = max_val + sum.ln();
    }
}
