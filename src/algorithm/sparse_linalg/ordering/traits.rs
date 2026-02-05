//! Traits for sparse matrix ordering algorithms.

use crate::error::Result;

/// Trait for sparse matrix ordering algorithms.
///
/// Ordering algorithms compute permutations that reduce fill-in during
/// sparse matrix factorization. The permutation is applied before
/// factorization: PAQ = LU (for non-symmetric) or PAP^T = LL^T (for symmetric).
pub trait SparseOrdering {
    /// Compute column permutation for a sparse matrix to minimize fill-in during LU factorization.
    ///
    /// Given a sparse matrix A in CSC format, computes a column permutation P such that
    /// factoring A*P (or equivalently, permuting columns) produces less fill-in than
    /// factoring A directly.
    ///
    /// # Arguments
    ///
    /// * `n_rows` - Number of rows in the matrix
    /// * `n_cols` - Number of columns in the matrix
    /// * `col_ptrs` - CSC column pointers (length n_cols + 1)
    /// * `row_indices` - CSC row indices
    ///
    /// # Returns
    ///
    /// Column permutation vector P where P[j] is the original column index
    /// that should be placed in position j of the permuted matrix.
    fn column_ordering(
        &self,
        n_rows: usize,
        n_cols: usize,
        col_ptrs: &[i64],
        row_indices: &[i64],
    ) -> Result<Vec<usize>>;

    /// Compute symmetric ordering for a sparse symmetric matrix.
    ///
    /// Given a sparse symmetric matrix A, computes a permutation P such that
    /// factoring PAP^T produces less fill-in. Only the lower triangular part
    /// is used.
    ///
    /// # Arguments
    ///
    /// * `n` - Matrix dimension (n x n)
    /// * `col_ptrs` - CSC column pointers for lower triangular part
    /// * `row_indices` - CSC row indices for lower triangular part
    ///
    /// # Returns
    ///
    /// Permutation vector P where P[i] is the original index that should
    /// be placed in position i of the permuted matrix.
    fn symmetric_ordering(
        &self,
        n: usize,
        col_ptrs: &[i64],
        row_indices: &[i64],
    ) -> Result<Vec<usize>>;
}
