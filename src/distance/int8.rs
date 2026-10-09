//! Signed 8-bit dot product, accumulated exactly and saturated to `i32`.

use super::kernels::Kernels;
use super::shape::{assert_same_len, for_each_row};
use crate::runtime::cpu::kernels::{i8xi8_dot_f32_with, i8xi8_dot_i32_with};

impl Kernels {
    /// i8 dot product `sum(a[i] * b[i])` at this handle's level.
    ///
    /// The total is exact. A total outside `i32` range saturates to
    /// `i32::MIN` or `i32::MAX`. Every level returns the same value.
    ///
    /// # Panics
    /// Panics when `a.len() != b.len()`.
    #[track_caller]
    #[inline]
    pub fn dot_i8(&self, a: &[i8], b: &[i8]) -> i32 {
        assert_same_len("dot_i8", a.len(), b.len());
        // SAFETY: `Kernels` only holds a level this CPU supports, and both
        // slices hold `a.len()` elements.
        unsafe { i8xi8_dot_i32_with(self.level(), a.as_ptr(), b.as_ptr(), a.len()) }
    }

    /// i8 dot product times `scale`: `(dot_i8(a, b) as f32) * scale`.
    ///
    /// # Panics
    /// Panics when `a.len() != b.len()`.
    #[track_caller]
    #[inline]
    pub fn dot_i8_scaled(&self, a: &[i8], b: &[i8], scale: f32) -> f32 {
        assert_same_len("dot_i8_scaled", a.len(), b.len());
        // SAFETY: `Kernels` only holds a level this CPU supports, and both
        // slices hold `a.len()` elements.
        unsafe { i8xi8_dot_f32_with(self.level(), a.as_ptr(), b.as_ptr(), scale, a.len()) }
    }

    /// i8 dot product of `query` with each row of `rows`, into `out`.
    ///
    /// `rows` is row-major with `d` components per row. `out[i]` equals
    /// `self.dot_i8(query, row_i)`.
    ///
    /// # Panics
    /// Panics when `query.len() != d` or `rows.len() != out.len() * d`.
    #[track_caller]
    pub fn dot_i8_many(&self, query: &[i8], rows: &[i8], d: usize, out: &mut [i32]) {
        for_each_row("dot_i8_many", query, rows, d, out, |row| {
            self.dot_i8(query, row)
        });
    }
}

/// i8 dot product `sum(a[i] * b[i])` at the best level this CPU supports.
///
/// The total is exact. A total outside `i32` range saturates to `i32::MIN` or
/// `i32::MAX`.
///
/// # Panics
/// Panics when `a.len() != b.len()`.
#[track_caller]
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    Kernels::detect().dot_i8(a, b)
}

/// i8 dot product times `scale`: `(dot_i8(a, b) as f32) * scale`.
///
/// # Panics
/// Panics when `a.len() != b.len()`.
#[track_caller]
#[inline]
pub fn dot_i8_scaled(a: &[i8], b: &[i8], scale: f32) -> f32 {
    Kernels::detect().dot_i8_scaled(a, b, scale)
}

/// i8 dot product of `query` with each row of `rows`, into `out`.
///
/// `rows` is row-major with `d` components per row. The level is detected once
/// per call.
///
/// # Panics
/// Panics when `query.len() != d` or `rows.len() != out.len() * d`.
#[track_caller]
pub fn dot_i8_many(query: &[i8], rows: &[i8], d: usize, out: &mut [i32]) {
    Kernels::detect().dot_i8_many(query, rows, d, out);
}
