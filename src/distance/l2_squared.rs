//! Squared Euclidean distance: `sum((a[i] - b[i])^2)`.

use super::kernels::Kernels;
use super::shape::{assert_same_len, for_each_row};
use crate::runtime::cpu::kernels::simd::distance as simd;

impl Kernels {
    /// f32 squared Euclidean distance `sum((a[i] - b[i])^2)` at this handle's level.
    ///
    /// # Panics
    /// Panics when `a.len() != b.len()`.
    #[track_caller]
    #[inline]
    pub fn l2_squared_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_same_len("l2_squared_f32", a.len(), b.len());
        // SAFETY: `Kernels` only holds a level this CPU supports, and both
        // slices hold `a.len()` elements.
        unsafe { simd::sqeuclidean_f32_with(self.level(), a.as_ptr(), b.as_ptr(), a.len()) }
    }

    /// f64 squared Euclidean distance `sum((a[i] - b[i])^2)` at this handle's level.
    ///
    /// # Panics
    /// Panics when `a.len() != b.len()`.
    #[track_caller]
    #[inline]
    pub fn l2_squared_f64(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_same_len("l2_squared_f64", a.len(), b.len());
        // SAFETY: `Kernels` only holds a level this CPU supports, and both
        // slices hold `a.len()` elements.
        unsafe { simd::sqeuclidean_f64_with(self.level(), a.as_ptr(), b.as_ptr(), a.len()) }
    }

    /// f32 squared Euclidean distance from `query` to each row of `rows`, into `out`.
    ///
    /// `rows` is row-major with `d` components per row. `out[i]` equals
    /// `self.l2_squared_f32(query, row_i)` bit for bit.
    ///
    /// # Panics
    /// Panics when `query.len() != d` or `rows.len() != out.len() * d`.
    #[track_caller]
    pub fn l2_squared_many_f32(&self, query: &[f32], rows: &[f32], d: usize, out: &mut [f32]) {
        for_each_row("l2_squared_many_f32", query, rows, d, out, |row| {
            self.l2_squared_f32(query, row)
        });
    }

    /// f64 squared Euclidean distance from `query` to each row of `rows`, into `out`.
    ///
    /// `rows` is row-major with `d` components per row. `out[i]` equals
    /// `self.l2_squared_f64(query, row_i)` bit for bit.
    ///
    /// # Panics
    /// Panics when `query.len() != d` or `rows.len() != out.len() * d`.
    #[track_caller]
    pub fn l2_squared_many_f64(&self, query: &[f64], rows: &[f64], d: usize, out: &mut [f64]) {
        for_each_row("l2_squared_many_f64", query, rows, d, out, |row| {
            self.l2_squared_f64(query, row)
        });
    }
}

/// f32 squared Euclidean distance `sum((a[i] - b[i])^2)` at the best level this CPU supports.
///
/// # Panics
/// Panics when `a.len() != b.len()`.
#[track_caller]
#[inline]
pub fn l2_squared_f32(a: &[f32], b: &[f32]) -> f32 {
    Kernels::detect().l2_squared_f32(a, b)
}

/// f64 squared Euclidean distance `sum((a[i] - b[i])^2)` at the best level this CPU supports.
///
/// # Panics
/// Panics when `a.len() != b.len()`.
#[track_caller]
#[inline]
pub fn l2_squared_f64(a: &[f64], b: &[f64]) -> f64 {
    Kernels::detect().l2_squared_f64(a, b)
}

/// f32 squared Euclidean distance from `query` to each row of `rows`, into `out`.
///
/// `rows` is row-major with `d` components per row. The level is detected once
/// per call.
///
/// # Panics
/// Panics when `query.len() != d` or `rows.len() != out.len() * d`.
#[track_caller]
pub fn l2_squared_many_f32(query: &[f32], rows: &[f32], d: usize, out: &mut [f32]) {
    Kernels::detect().l2_squared_many_f32(query, rows, d, out);
}

/// f64 squared Euclidean distance from `query` to each row of `rows`, into `out`.
///
/// `rows` is row-major with `d` components per row. The level is detected once
/// per call.
///
/// # Panics
/// Panics when `query.len() != d` or `rows.len() != out.len() * d`.
#[track_caller]
pub fn l2_squared_many_f64(query: &[f64], rows: &[f64], d: usize, out: &mut [f64]) {
    Kernels::detect().l2_squared_many_f64(query, rows, d, out);
}
