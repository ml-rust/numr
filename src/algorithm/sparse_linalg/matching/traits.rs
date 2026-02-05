//! Traits for bipartite matching algorithms.

use crate::error::Result;

/// Trait for bipartite matching algorithms on sparse matrices.
///
/// These algorithms find maximum matchings in the bipartite graph formed by
/// the row-column structure of a sparse matrix. A matching pairs rows with
/// columns such that each row and column appears in at most one pair.
pub trait BipartiteMatching {
    /// Compute maximum transversal (row permutation for diagonal nonzeros).
    ///
    /// Given a sparse matrix A, finds a row permutation P such that P*A has
    /// maximum number of nonzeros on the diagonal. This is equivalent to finding
    /// a maximum matching in the bipartite graph where rows and columns are
    /// vertices and nonzeros are edges.
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
    /// Tuple of (row_perm, col_match) where:
    /// - `row_perm[i]` = row index that should be in position i (or -1 if unmatched)
    /// - `col_match[j]` = row matched to column j (or -1 if unmatched)
    fn maximum_transversal(
        &self,
        n_rows: usize,
        n_cols: usize,
        col_ptrs: &[i64],
        row_indices: &[i64],
    ) -> Result<(Vec<i32>, Vec<i32>)>;

    /// Compute structural rank of a sparse matrix.
    ///
    /// The structural rank is the maximum number of nonzeros that can be placed
    /// on the diagonal by row and column permutations. It equals the size of
    /// the maximum matching.
    ///
    /// # Arguments
    ///
    /// * `n_rows` - Number of rows in the matrix
    /// * `n_cols` - Number of columns in the matrix
    /// * `col_ptrs` - CSC column pointers
    /// * `row_indices` - CSC row indices
    ///
    /// # Returns
    ///
    /// The structural rank (0 to min(n_rows, n_cols)).
    fn structural_rank(
        &self,
        n_rows: usize,
        n_cols: usize,
        col_ptrs: &[i64],
        row_indices: &[i64],
    ) -> Result<usize>;
}
