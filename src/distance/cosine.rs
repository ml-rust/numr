//! Cosine distance: `1 - dot(a, b) / (|a| * |b|)`.
//!
//! The kernel gathers the three sums in one pass. `metrics::cosine_from_sums`
//! turns them into a distance, the same formula `cdist` and `pdist` use. A
//! zero denominator gives 0.

use super::kernels::Kernels;
use super::shape::{assert_same_len, for_each_row};
use crate::runtime::cpu::kernels::distance::metrics::cosine_from_sums;
use crate::runtime::cpu::kernels::simd::distance as simd;

impl Kernels {
    /// f32 cosine distance `1 - dot(a, b) / (|a| * |b|)` at this
    /// handle's level.
    ///
    /// Returns 0 when either vector is all zeros, and for empty vectors.
    ///
    /// # Panics
    /// Panics when `a.len() != b.len()`.
    #[track_caller]
    #[inline]
    pub fn cosine_distance_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_same_len("cosine_distance_f32", a.len(), b.len());
        // SAFETY: `Kernels` only holds a level this CPU supports, and both
        // slices hold `a.len()` elements.
        let s =
            unsafe { simd::cosine_sums_f32_with(self.level(), a.as_ptr(), b.as_ptr(), a.len()) };
        cosine_from_sums(s.dot, s.norm_a, s.norm_b)
    }

    /// f64 cosine distance `1 - dot(a, b) / (|a| * |b|)` at this
    /// handle's level.
    ///
    /// Returns 0 when either vector is all zeros, and for empty vectors.
    ///
    /// # Panics
    /// Panics when `a.len() != b.len()`.
    #[track_caller]
    #[inline]
    pub fn cosine_distance_f64(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_same_len("cosine_distance_f64", a.len(), b.len());
        // SAFETY: `Kernels` only holds a level this CPU supports, and both
        // slices hold `a.len()` elements.
        let s =
            unsafe { simd::cosine_sums_f64_with(self.level(), a.as_ptr(), b.as_ptr(), a.len()) };
        cosine_from_sums(s.dot, s.norm_a, s.norm_b)
    }

    /// f32 cosine distance from `query` to each row of `rows`, into `out`.
    ///
    /// `rows` is row-major with `d` components per row. `out[i]` equals
    /// `self.cosine_distance_f32(query, row_i)` bit for bit.
    ///
    /// # Panics
    /// Panics when `query.len() != d` or `rows.len() != out.len() * d`.
    #[track_caller]
    pub fn cosine_distance_many_f32(&self, query: &[f32], rows: &[f32], d: usize, out: &mut [f32]) {
        // Each row runs the one-pass cosine sums, which recompute `|query|^2`.
        // A query norm taken once from the dot kernel sums in a different
        // order, so `out[i]` would no longer match the per-pair call bit for bit.
        for_each_row("cosine_distance_many_f32", query, rows, d, out, |row| {
            self.cosine_distance_f32(query, row)
        });
    }

    /// f64 cosine distance from `query` to each row of `rows`, into `out`.
    ///
    /// `rows` is row-major with `d` components per row. `out[i]` equals
    /// `self.cosine_distance_f64(query, row_i)` bit for bit.
    ///
    /// # Panics
    /// Panics when `query.len() != d` or `rows.len() != out.len() * d`.
    #[track_caller]
    pub fn cosine_distance_many_f64(&self, query: &[f64], rows: &[f64], d: usize, out: &mut [f64]) {
        // Per-row cosine sums, for the reason in `cosine_distance_many_f32`.
        for_each_row("cosine_distance_many_f64", query, rows, d, out, |row| {
            self.cosine_distance_f64(query, row)
        });
    }
}

/// f32 cosine distance `1 - dot(a, b) / (|a| * |b|)` at the best
/// level this CPU supports.
///
/// Returns 0 when either vector is all zeros, and for empty vectors.
///
/// # Panics
/// Panics when `a.len() != b.len()`.
#[track_caller]
#[inline]
pub fn cosine_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    Kernels::detect().cosine_distance_f32(a, b)
}

/// f64 cosine distance `1 - dot(a, b) / (|a| * |b|)` at the best
/// level this CPU supports.
///
/// Returns 0 when either vector is all zeros, and for empty vectors.
///
/// # Panics
/// Panics when `a.len() != b.len()`.
#[track_caller]
#[inline]
pub fn cosine_distance_f64(a: &[f64], b: &[f64]) -> f64 {
    Kernels::detect().cosine_distance_f64(a, b)
}

/// f32 cosine distance from `query` to each row of `rows`, into `out`.
///
/// `rows` is row-major with `d` components per row. The level is detected once
/// per call. For unit-length rows and query, [`dot_many_f32`](super::dot_many_f32)
/// gives `1 - distance` with one sum per row instead of three.
///
/// # Panics
/// Panics when `query.len() != d` or `rows.len() != out.len() * d`.
#[track_caller]
pub fn cosine_distance_many_f32(query: &[f32], rows: &[f32], d: usize, out: &mut [f32]) {
    Kernels::detect().cosine_distance_many_f32(query, rows, d, out);
}

/// f64 cosine distance from `query` to each row of `rows`, into `out`.
///
/// `rows` is row-major with `d` components per row. The level is detected once
/// per call. For unit-length rows and query, [`dot_many_f64`](super::dot_many_f64)
/// gives `1 - distance` with one sum per row instead of three.
///
/// # Panics
/// Panics when `query.len() != d` or `rows.len() != out.len() * d`.
#[track_caller]
pub fn cosine_distance_many_f64(query: &[f64], rows: &[f64], d: usize, out: &mut [f64]) {
    Kernels::detect().cosine_distance_many_f64(query, rows, d, out);
}
