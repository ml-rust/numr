//! Scalar dot product.

use num_traits::Float;

/// `sum(a[i] * b[i])` with one accumulator, in index order.
///
/// # Safety
/// `a` and `b` must each be valid for `len` reads.
#[inline]
unsafe fn dot<F: Float>(a: *const F, b: *const F, len: usize) -> F {
    let mut sum = F::zero();
    for k in 0..len {
        sum = sum + *a.add(k) * *b.add(k);
    }
    sum
}

/// Scalar f32 dot product.
///
/// # Safety
/// `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn dot_f32(a: *const f32, b: *const f32, len: usize) -> f32 {
    dot(a, b, len)
}

/// Scalar f64 dot product.
///
/// # Safety
/// `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn dot_f64(a: *const f64, b: *const f64, len: usize) -> f64 {
    dot(a, b, len)
}
