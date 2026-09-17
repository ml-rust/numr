//! Scalar fallback for softmax backward.

/// Scalar softmax backward for f32
#[inline]
pub(super) unsafe fn softmax_bwd_scalar_f32(
    grad: *const f32,
    output: *const f32,
    d_input: *mut f32,
    outer_size: usize,
    dim_size: usize,
) {
    for o in 0..outer_size {
        let base = o * dim_size;

        // Pass 1: dot = sum(grad * output)
        let mut dot = 0.0f32;
        for d in 0..dim_size {
            dot += *grad.add(base + d) * *output.add(base + d);
        }

        // Pass 2: d_input = output * (grad - dot)
        for d in 0..dim_size {
            let idx = base + d;
            *d_input.add(idx) = *output.add(idx) * (*grad.add(idx) - dot);
        }
    }
}

/// Scalar softmax backward for f64
#[inline]
pub(super) unsafe fn softmax_bwd_scalar_f64(
    grad: *const f64,
    output: *const f64,
    d_input: *mut f64,
    outer_size: usize,
    dim_size: usize,
) {
    for o in 0..outer_size {
        let base = o * dim_size;

        // Pass 1: dot = sum(grad * output)
        let mut dot = 0.0f64;
        for d in 0..dim_size {
            dot += *grad.add(base + d) * *output.add(base + d);
        }

        // Pass 2: d_input = output * (grad - dot)
        for d in 0..dim_size {
            let idx = base + d;
            *d_input.add(idx) = *output.add(idx) * (*grad.add(idx) - dot);
        }
    }
}
