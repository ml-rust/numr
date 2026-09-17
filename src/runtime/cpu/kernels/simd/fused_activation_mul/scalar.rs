//! Scalar fallbacks for fused activation-multiplication operations.

/// Scalar silu_mul for f32
#[inline]
pub unsafe fn silu_mul_scalar_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        *out.add(i) = (x / (1.0 + (-x).exp())) * y;
    }
}

/// Scalar silu_mul for f64
#[inline]
pub unsafe fn silu_mul_scalar_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        *out.add(i) = (x / (1.0 + (-x).exp())) * y;
    }
}

/// Scalar gelu_mul for f32
#[inline]
pub unsafe fn gelu_mul_scalar_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    const TANH_COEF: f32 = 0.044715;

    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        let inner = SQRT_2_OVER_PI * (x + TANH_COEF * x * x * x);
        *out.add(i) = 0.5 * x * (1.0 + inner.tanh()) * y;
    }
}

/// Scalar gelu_mul for f64
#[inline]
pub unsafe fn gelu_mul_scalar_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
    const TANH_COEF: f64 = 0.044715;

    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        let inner = SQRT_2_OVER_PI * (x + TANH_COEF * x * x * x);
        *out.add(i) = 0.5 * x * (1.0 + inner.tanh()) * y;
    }
}

/// Scalar relu_mul for f32
#[inline]
pub unsafe fn relu_mul_scalar_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        *out.add(i) = if x > 0.0 { x * y } else { 0.0 };
    }
}

/// Scalar relu_mul for f64
#[inline]
pub unsafe fn relu_mul_scalar_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        *out.add(i) = if x > 0.0 { x * y } else { 0.0 };
    }
}

/// Scalar sigmoid_mul for f32
#[inline]
pub unsafe fn sigmoid_mul_scalar_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        *out.add(i) = (1.0 / (1.0 + (-x).exp())) * y;
    }
}

/// Scalar sigmoid_mul for f64
#[inline]
pub unsafe fn sigmoid_mul_scalar_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    for i in 0..len {
        let x = *a.add(i);
        let y = *b.add(i);
        *out.add(i) = (1.0 / (1.0 + (-x).exp())) * y;
    }
}
