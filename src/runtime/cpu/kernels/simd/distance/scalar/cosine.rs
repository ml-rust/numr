//! Scalar one-pass cosine sums.

use super::super::sums::CosineSums;
use num_traits::Float;

/// `a·b`, `a·a` and `b·b` with one accumulator each, in index order.
///
/// # Safety
/// `a` and `b` must each be valid for `len` reads.
#[inline]
unsafe fn cosine_sums<F: Float>(a: *const F, b: *const F, len: usize) -> CosineSums<F> {
    let mut dot = F::zero();
    let mut norm_a = F::zero();
    let mut norm_b = F::zero();
    for k in 0..len {
        let ak = *a.add(k);
        let bk = *b.add(k);
        dot = dot + ak * bk;
        norm_a = norm_a + ak * ak;
        norm_b = norm_b + bk * bk;
    }
    CosineSums {
        dot,
        norm_a,
        norm_b,
    }
}

/// Scalar f32 cosine sums.
///
/// # Safety
/// `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn cosine_sums_f32(a: *const f32, b: *const f32, len: usize) -> CosineSums<f32> {
    cosine_sums(a, b, len)
}

/// Scalar f64 cosine sums.
///
/// # Safety
/// `a` and `b` must each be valid for `len` reads.
#[inline]
pub unsafe fn cosine_sums_f64(a: *const f64, b: *const f64, len: usize) -> CosineSums<f64> {
    cosine_sums(a, b, len)
}
