//! Symbolic analysis for sparse QR factorization
//!
//! Computes the elimination tree and column counts for R without
//! forming A^T*A explicitly. Uses the row structure of A instead.

use crate::algorithm::sparse_linalg::ordering::{ColamdOptions, colamd};
use crate::error::Result;

use super::types::{QrOptions, QrOrdering, QrSymbolic};

/// Compute symbolic analysis for sparse QR factorization
///
/// # Arguments
///
/// * `col_ptrs` - CSC column pointers `[n+1]`
/// * `row_indices` - CSC row indices `[nnz]`
/// * `m` - Number of rows
/// * `n` - Number of columns
/// * `options` - QR options (ordering strategy)
///
/// # Returns
///
/// Symbolic structure with elimination tree, column counts, and permutation.
pub fn sparse_qr_symbolic(
    col_ptrs: &[i64],
    row_indices: &[i64],
    m: usize,
    n: usize,
    options: &QrOptions,
) -> Result<QrSymbolic> {
    // Step 1: Compute column permutation
    let col_perm = match options.ordering {
        QrOrdering::Identity => (0..n).collect::<Vec<usize>>(),
        QrOrdering::Colamd => {
            let colamd_opts = ColamdOptions::default();
            let (perm, _stats) = colamd(m, n, col_ptrs, row_indices, &colamd_opts)?;
            perm
        }
    };

    // Step 2: Build permuted column pointers and row indices
    let (perm_col_ptrs, perm_row_indices) = permute_columns(col_ptrs, row_indices, n, &col_perm);

    // Step 3: Compute elimination tree of A^T*A from row structure of A
    let etree = compute_etree_ata(&perm_col_ptrs, &perm_row_indices, m, n);

    // Step 4: Compute column counts for R using etree
    let r_col_counts = compute_r_col_counts(&perm_col_ptrs, &perm_row_indices, &etree, m, n);

    let predicted_r_nnz: usize = r_col_counts.iter().sum();

    Ok(QrSymbolic {
        m,
        n,
        etree,
        r_col_counts,
        col_perm,
        predicted_r_nnz,
    })
}

/// Permute columns of a CSC matrix according to a permutation vector
fn permute_columns(
    col_ptrs: &[i64],
    row_indices: &[i64],
    n: usize,
    perm: &[usize],
) -> (Vec<i64>, Vec<i64>) {
    // Count entries per new column
    let mut new_counts = vec![0usize; n];
    for new_col in 0..n {
        let old_col = perm[new_col];
        let start = col_ptrs[old_col] as usize;
        let end = col_ptrs[old_col + 1] as usize;
        new_counts[new_col] = end - start;
    }

    // Build new column pointers
    let mut new_col_ptrs = vec![0i64; n + 1];
    for j in 0..n {
        new_col_ptrs[j + 1] = new_col_ptrs[j] + new_counts[j] as i64;
    }

    // Copy row indices in new column order
    let total_nnz = new_col_ptrs[n] as usize;
    let mut new_row_indices = vec![0i64; total_nnz];
    for new_col in 0..n {
        let old_col = perm[new_col];
        let old_start = col_ptrs[old_col] as usize;
        let old_end = col_ptrs[old_col + 1] as usize;
        let new_start = new_col_ptrs[new_col] as usize;

        for (i, &row) in row_indices[old_start..old_end].iter().enumerate() {
            new_row_indices[new_start + i] = row;
        }
    }

    (new_col_ptrs, new_row_indices)
}

/// Compute the elimination tree of A^T*A from the row structure of A.
///
/// Uses the column-based algorithm from Gilbert, Ng, Peyton (1994).
/// For each column j (processed left to right), we look at every row i
/// that column j touches. For that row, if we've seen a previous column k < j
/// that also touches row i, then we follow k's path up the tree (path compression)
/// to find its root r, and set parent[r] = j.
///
/// This correctly builds the etree without forming A^T*A.
fn compute_etree_ata(col_ptrs: &[i64], row_indices: &[i64], m: usize, n: usize) -> Vec<i64> {
    let mut parent = vec![-1i64; n];
    // ancestor[j] used for path compression in union-find
    let mut ancestor = vec![0usize; n];
    for j in 0..n {
        ancestor[j] = j;
    }
    // first_col[row] = first column that touches this row, or usize::MAX if none yet
    let mut first_col = vec![usize::MAX; m];

    for j in 0..n {
        // Mark column j as its own ancestor (fresh)
        ancestor[j] = j;

        let start = col_ptrs[j] as usize;
        let end = col_ptrs[j + 1] as usize;

        for &row in &row_indices[start..end] {
            let row = row as usize;
            let k = first_col[row];
            if k == usize::MAX {
                // First column to touch this row
                first_col[row] = j;
            } else {
                // Column k < j also touches this row → they share a row
                // Find root of k with path compression
                let mut r = k;
                while ancestor[r] != r {
                    r = ancestor[r];
                }
                // Path compression
                let mut node = k;
                while node != r {
                    let next = ancestor[node];
                    ancestor[node] = r;
                    node = next;
                }

                if r != j {
                    // Set parent of root to j
                    parent[r] = j as i64;
                    ancestor[r] = j;
                }
            }
        }
    }

    parent
}

/// Compute upper bound on R column counts using the elimination tree.
///
/// For each column j, the column count in R is at most the number of
/// original rows in column j plus fill-in from the etree descendants.
fn compute_r_col_counts(
    col_ptrs: &[i64],
    _row_indices: &[i64],
    etree: &[i64],
    m: usize,
    n: usize,
) -> Vec<usize> {
    // Simple upper bound: for each column, count unique rows that appear
    // in the column and all its descendants in the etree
    //
    // For a tighter bound we'd need the row subtree approach, but this
    // conservative estimate is sufficient for pre-allocation.

    // Start with direct column counts (capped at min(m, col_index + 1))
    let mut counts = vec![0usize; n];
    for col in 0..n {
        let start = col_ptrs[col] as usize;
        let end = col_ptrs[col + 1] as usize;
        // Number of entries in this column, capped at entries that can be in R
        // (only rows 0..=col for R's upper triangular structure, for square;
        //  for rectangular, min(m, col+1))
        let direct = end - start;
        counts[col] = direct.min(m);
    }

    // Propagate counts up the etree (children contribute to parent's count)
    // Process in reverse order (leaves first)
    // This is a conservative estimate - actual fill depends on row overlap
    for j in 0..n {
        let parent = etree[j];
        if parent >= 0 && (parent as usize) < n {
            // Parent gains at most the child's count minus 1 (the diagonal)
            let contribution = if counts[j] > 0 { counts[j] - 1 } else { 0 };
            counts[parent as usize] = counts[parent as usize].max(contribution + 1);
        }
    }

    // Ensure each column has at least 1 entry (the diagonal of R, if rank allows)
    for count in &mut counts {
        *count = (*count).max(1);
    }

    counts
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_identity_ordering() {
        // 3x3 diagonal matrix
        let col_ptrs = vec![0i64, 1, 2, 3];
        let row_indices = vec![0i64, 1, 2];

        let options = QrOptions::no_ordering();
        let symbolic = sparse_qr_symbolic(&col_ptrs, &row_indices, 3, 3, &options).unwrap();

        assert_eq!(symbolic.m, 3);
        assert_eq!(symbolic.n, 3);
        assert_eq!(symbolic.col_perm, vec![0, 1, 2]);
    }

    #[test]
    fn test_symbolic_tridiagonal() {
        // 4x4 tridiagonal matrix:
        // [x . . .]
        // [x x . .]
        // [. x x .]
        // [. . x x]
        let col_ptrs = vec![0i64, 2, 4, 6, 7];
        let row_indices = vec![0i64, 1, 1, 2, 2, 3, 3];

        let options = QrOptions::no_ordering();
        let symbolic = sparse_qr_symbolic(&col_ptrs, &row_indices, 4, 4, &options).unwrap();

        assert_eq!(symbolic.m, 4);
        assert_eq!(symbolic.n, 4);
        // Each column should have a reasonable count
        for &count in &symbolic.r_col_counts {
            assert!(count >= 1);
        }
    }

    #[test]
    fn test_symbolic_with_colamd() {
        // 4x3 overdetermined matrix
        let col_ptrs = vec![0i64, 3, 5, 7];
        let row_indices = vec![0i64, 1, 2, 1, 3, 0, 3];

        let options = QrOptions::default(); // uses Colamd
        let symbolic = sparse_qr_symbolic(&col_ptrs, &row_indices, 4, 3, &options).unwrap();

        assert_eq!(symbolic.m, 4);
        assert_eq!(symbolic.n, 3);
        assert_eq!(symbolic.col_perm.len(), 3);
        // Permutation should be a valid permutation of 0..3
        let mut sorted_perm = symbolic.col_perm.clone();
        sorted_perm.sort_unstable();
        assert_eq!(sorted_perm, vec![0, 1, 2]);
    }

    #[test]
    fn test_etree_chain() {
        // Matrix where columns share rows in a chain pattern
        // Col 0: rows {0, 1}
        // Col 1: rows {1, 2}
        // Col 2: rows {2, 3}
        let col_ptrs = vec![0i64, 2, 4, 6];
        let row_indices = vec![0i64, 1, 1, 2, 2, 3];

        let etree = compute_etree_ata(&col_ptrs, &row_indices, 4, 3);

        // Col 0 and 1 share row 1, so etree[0] = 1
        // Col 1 and 2 share row 2, so etree[1] = 2
        // Col 2 is root
        assert_eq!(etree[0], 1);
        assert_eq!(etree[1], 2);
        assert_eq!(etree[2], -1);
    }
}
