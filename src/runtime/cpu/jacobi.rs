//! Shared Jacobi algorithm utilities for SVD and Eigendecomposition
//!
//! This module contains the common Jacobi rotation computation and matrix
//! operations shared between One-Sided Jacobi SVD and Two-Sided Jacobi
//! Eigendecomposition. The rotation parameters use the numerically stable
//! LAPACK formula to avoid catastrophic cancellation.

use crate::dtype::Element;

/// Trait for elements that support linear algebra operations.
///
/// This trait extends `Element` with operations needed for numerical
/// linear algebra algorithms. Methods like `zero()`, `one()`, `to_f64()`,
/// and `from_f64()` are inherited from `Element`.
pub trait LinalgElement: Element + Sized {
    /// Returns machine epsilon for this type
    fn epsilon_val() -> f64;
    /// Returns absolute value
    fn abs_val(&self) -> Self;
    /// Returns square root
    fn sqrt_val(&self) -> Self;
    /// Returns negation
    fn neg_val(&self) -> Self;
}

impl LinalgElement for f32 {
    #[inline]
    fn epsilon_val() -> f64 {
        f32::EPSILON as f64
    }
    #[inline]
    fn abs_val(&self) -> Self {
        self.abs()
    }
    #[inline]
    fn sqrt_val(&self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn neg_val(&self) -> Self {
        -*self
    }
}

impl LinalgElement for f64 {
    #[inline]
    fn epsilon_val() -> f64 {
        f64::EPSILON
    }
    #[inline]
    fn abs_val(&self) -> Self {
        self.abs()
    }
    #[inline]
    fn sqrt_val(&self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn neg_val(&self) -> Self {
        -*self
    }
}

/// Jacobi rotation parameters (cosine and sine of rotation angle).
///
/// These parameters define a Givens rotation matrix:
/// ```text
/// J = [ c  -s ]
///     [ s   c ]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct JacobiRotation {
    /// Cosine of rotation angle
    pub c: f64,
    /// Sine of rotation angle
    pub s: f64,
}

impl JacobiRotation {
    /// Compute Jacobi rotation parameters using numerically stable LAPACK formula.
    ///
    /// Given 2x2 symmetric submatrix elements, computes rotation that zeroes
    /// the off-diagonal element.
    ///
    /// # Algorithm
    /// ```text
    /// τ = (a_qq - a_pp) / (2 * a_pq)
    /// t = sign(τ) / (|τ| + sqrt(1 + τ²))
    /// c = 1 / sqrt(1 + t²)
    /// s = t * c
    /// ```
    ///
    /// # Arguments
    /// * `a_pp` - Diagonal element at (p, p)
    /// * `a_qq` - Diagonal element at (q, q)
    /// * `a_pq` - Off-diagonal element at (p, q)
    #[inline]
    pub fn compute(a_pp: f64, a_qq: f64, a_pq: f64) -> Self {
        let tau_num = a_qq - a_pp;
        let tau_den = 2.0 * a_pq;

        if tau_den.abs() < 1e-300 {
            return Self { c: 1.0, s: 0.0 };
        }

        let tau = tau_num / tau_den;
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };

        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        Self { c, s }
    }

    /// Returns typed rotation parameters.
    #[inline]
    pub fn typed<T: LinalgElement>(&self) -> (T, T) {
        (T::from_f64(self.c), T::from_f64(self.s))
    }
}

/// Apply Jacobi rotation to two columns of a matrix.
///
/// Computes: `[col_p', col_q'] = [col_p, col_q] @ [[c, s], [-s, c]]`
///
/// # Arguments
/// * `data` - Matrix data in row-major order [rows × cols]
/// * `rows` - Number of rows
/// * `cols` - Number of columns (stride)
/// * `p`, `q` - Column indices
/// * `rot` - Rotation parameters
#[inline]
pub fn apply_rotation_to_columns<T: LinalgElement>(
    data: &mut [T],
    rows: usize,
    cols: usize,
    p: usize,
    q: usize,
    rot: &JacobiRotation,
) {
    let (c, s): (T, T) = rot.typed();

    for i in 0..rows {
        let idx_p = i * cols + p;
        let idx_q = i * cols + q;

        let val_p = data[idx_p];
        let val_q = data[idx_q];

        data[idx_p] = c * val_p - s * val_q;
        data[idx_q] = s * val_p + c * val_q;
    }
}

/// Apply two-sided Jacobi rotation to a symmetric matrix.
///
/// Computes: `A' = J^T @ A @ J` which zeroes out `A[p,q]` and `A[q,p]`.
///
/// # Arguments
/// * `work` - Symmetric matrix [n × n] in row-major order
/// * `n` - Matrix dimension
/// * `p`, `q` - Element indices (p < q)
/// * `rot` - Rotation parameters
/// * `a_pp`, `a_qq`, `a_pq` - Original matrix elements
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn apply_two_sided_rotation<T: LinalgElement>(
    work: &mut [T],
    n: usize,
    p: usize,
    q: usize,
    rot: &JacobiRotation,
    a_pp: T,
    a_qq: T,
    a_pq: T,
) {
    let (c, s): (T, T) = rot.typed();

    // Update off-diagonal rows/columns
    for k in 0..n {
        if k != p && k != q {
            let a_kp = work[k * n + p];
            let a_kq = work[k * n + q];

            let new_kp = c * a_kp - s * a_kq;
            let new_kq = s * a_kp + c * a_kq;

            work[k * n + p] = new_kp;
            work[p * n + k] = new_kp;
            work[k * n + q] = new_kq;
            work[q * n + k] = new_kq;
        }
    }

    // Update diagonal elements
    let c2 = T::from_f64(rot.c * rot.c);
    let s2 = T::from_f64(rot.s * rot.s);
    let cs2 = T::from_f64(2.0 * rot.c * rot.s);

    work[p * n + p] = c2 * a_pp - cs2 * a_pq + s2 * a_qq;
    work[q * n + q] = s2 * a_pp + cs2 * a_pq + c2 * a_qq;
    work[p * n + q] = T::zero();
    work[q * n + p] = T::zero();
}

/// Compute Gram matrix elements for two columns (for One-Sided Jacobi SVD).
///
/// Returns `(a_pp, a_qq, a_pq)` where:
/// - `a_pp = B[:,p] · B[:,p]`
/// - `a_qq = B[:,q] · B[:,q]`
/// - `a_pq = B[:,p] · B[:,q]`
#[inline]
pub fn compute_gram_elements<T: LinalgElement>(
    b: &[T],
    rows: usize,
    cols: usize,
    p: usize,
    q: usize,
) -> (T, T, T) {
    let mut a_pp = T::zero();
    let mut a_qq = T::zero();
    let mut a_pq = T::zero();

    for i in 0..rows {
        let bp = b[i * cols + p];
        let bq = b[i * cols + q];
        a_pp = a_pp + bp * bp;
        a_qq = a_qq + bq * bq;
        a_pq = a_pq + bp * bq;
    }

    (a_pp, a_qq, a_pq)
}

/// Sort indices by value magnitude (descending).
#[inline]
pub fn argsort_by_magnitude_desc<T: LinalgElement>(values: &[T]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&i, &j| {
        values[j]
            .abs_val()
            .to_f64()
            .partial_cmp(&values[i].abs_val().to_f64())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

/// Sort indices by value (descending).
#[inline]
pub fn argsort_desc<T: LinalgElement>(values: &[T]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&i, &j| {
        values[j]
            .to_f64()
            .partial_cmp(&values[i].to_f64())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

/// Reorder vector elements according to index permutation.
#[inline]
pub fn permute_vector<T: LinalgElement>(data: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&idx| data[idx]).collect()
}

/// Reorder matrix columns according to index permutation.
#[inline]
pub fn permute_columns<T: LinalgElement>(
    data: &[T],
    rows: usize,
    cols: usize,
    indices: &[usize],
    new_cols: usize,
) -> Vec<T> {
    let mut result = vec![T::zero(); rows * new_cols];
    for (new_idx, &old_idx) in indices.iter().take(new_cols).enumerate() {
        for i in 0..rows {
            result[i * new_cols + new_idx] = data[i * cols + old_idx];
        }
    }
    result
}

/// Initialize an identity matrix [n × n].
#[inline]
pub fn identity_matrix<T: LinalgElement>(n: usize) -> Vec<T> {
    let mut result = vec![T::zero(); n * n];
    for i in 0..n {
        result[i * n + i] = T::one();
    }
    result
}

/// Compute column norms and normalize columns in-place.
///
/// Returns vector of column norms.
#[inline]
pub fn normalize_columns<T: LinalgElement>(
    data: &mut [T],
    rows: usize,
    cols: usize,
    eps: f64,
) -> Vec<T> {
    let mut norms = vec![T::zero(); cols];

    for j in 0..cols {
        let mut norm_sq = T::zero();
        for i in 0..rows {
            let val = data[i * cols + j];
            norm_sq = norm_sq + val * val;
        }
        let norm = norm_sq.sqrt_val();
        norms[j] = norm;

        if norm.to_f64() > eps {
            for i in 0..rows {
                data[i * cols + j] = data[i * cols + j] / norm;
            }
        } else {
            for i in 0..rows {
                data[i * cols + j] = T::zero();
            }
        }
    }

    norms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jacobi_rotation_zero_offdiag() {
        let rot = JacobiRotation::compute(1.0, 2.0, 0.0);
        assert!((rot.c - 1.0).abs() < 1e-10);
        assert!(rot.s.abs() < 1e-10);
    }

    #[test]
    fn test_jacobi_rotation_equal_diag() {
        let rot = JacobiRotation::compute(1.0, 1.0, 0.5);
        let expected = 1.0 / 2.0f64.sqrt();
        assert!((rot.c - expected).abs() < 1e-10);
        assert!((rot.s.abs() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_argsort_by_magnitude() {
        let values: Vec<f64> = vec![1.0, -3.0, 2.0, -0.5];
        let indices = argsort_by_magnitude_desc(&values);
        assert_eq!(indices, vec![1, 2, 0, 3]);
    }
}
