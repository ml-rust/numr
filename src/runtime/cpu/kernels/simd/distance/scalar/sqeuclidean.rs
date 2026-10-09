//! Scalar squared Euclidean distance.

use num_traits::Float;

/// `sum((a[i] - b[i])^2)` with one accumulator, in index order.
///
/// The difference is squared with a separate multiply and add, never a fused
/// multiply-add. That matches the generic loop in `metrics::sqeuclidean`.
///
/// # Safety
/// `a` and `b` must each be valid for `len` reads.
#[inline]
unsafe fn sqeuclidean<F: Float>(a: *const F, b: *const F, len: usize) -> F {
    let mut sum = F::zero();
    for k in 0..len {
        let diff = *a.add(k) - *b.add(k);
        sum = sum + diff * diff;
    }
    sum
}

/// Scalar f32 squared Euclidean distance.
///
/// # Safety
/// `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn sqeuclidean_f32(a: *const f32, b: *const f32, len: usize) -> f32 {
    sqeuclidean(a, b, len)
}

/// Scalar f64 squared Euclidean distance.
///
/// # Safety
/// `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn sqeuclidean_f64(a: *const f64, b: *const f64, len: usize) -> f64 {
    sqeuclidean(a, b, len)
}
