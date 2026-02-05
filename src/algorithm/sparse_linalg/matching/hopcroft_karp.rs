//! Hopcroft-Karp algorithm for maximum bipartite matching.
//!
//! Finds a maximum cardinality matching in a bipartite graph in O(E√V) time.
//! Used for computing maximum transversal in sparse matrices.
//!
//! # Algorithm Overview
//!
//! The algorithm alternates between two phases:
//!
//! 1. **BFS phase**: Find shortest augmenting paths from all unmatched columns
//!    to unmatched rows, building a level graph.
//!
//! 2. **DFS phase**: Find a maximal set of vertex-disjoint augmenting paths
//!    in the level graph and augment the matching.
//!
//! This is repeated until no augmenting paths exist.
//!
//! # Complexity
//!
//! - Time: O(E√V) where E = nnz, V = n_rows + n_cols
//! - Space: O(V) for level arrays and matching arrays
//!
//! # References
//!
//! Hopcroft, J. E., & Karp, R. M. (1973). "An n^(5/2) algorithm for maximum
//! matchings in bipartite graphs." SIAM Journal on Computing, 2(4), 225-231.

use std::collections::VecDeque;

use crate::error::{Error, Result};

/// Result of maximum matching computation.
#[derive(Debug, Clone)]
pub struct MatchingResult {
    /// For each column j, row_match[j] = row matched to column j, or -1 if unmatched.
    pub col_to_row: Vec<i32>,
    /// For each row i, col_match[i] = column matched to row i, or -1 if unmatched.
    pub row_to_col: Vec<i32>,
    /// Size of the maximum matching (number of matched pairs).
    pub matching_size: usize,
    /// Structural rank (same as matching_size for square matrices).
    pub structural_rank: usize,
}

/// Sentinel value for unmatched vertices and infinite distance.
const NIL: i32 = -1;
const INF: i32 = i32::MAX;

/// Compute maximum bipartite matching using Hopcroft-Karp algorithm.
///
/// The bipartite graph has columns on one side (U) and rows on the other (V).
/// An edge exists between column j and row i if the matrix has a nonzero at (i, j).
///
/// # Arguments
///
/// * `n_rows` - Number of rows (|V|)
/// * `n_cols` - Number of columns (|U|)
/// * `col_ptrs` - CSC column pointers (length n_cols + 1)
/// * `row_indices` - CSC row indices (edges from columns to rows)
///
/// # Returns
///
/// MatchingResult containing the maximum matching.
pub fn hopcroft_karp(
    n_rows: usize,
    n_cols: usize,
    col_ptrs: &[i64],
    row_indices: &[i64],
) -> Result<MatchingResult> {
    // Validate inputs
    if col_ptrs.len() != n_cols + 1 {
        return Err(Error::InvalidArgument {
            arg: "col_ptrs",
            reason: format!(
                "length {} does not match n_cols + 1 = {}",
                col_ptrs.len(),
                n_cols + 1
            ),
        });
    }

    let nnz = col_ptrs[n_cols] as usize;
    if row_indices.len() < nnz {
        return Err(Error::InvalidArgument {
            arg: "row_indices",
            reason: format!("length {} is less than nnz = {}", row_indices.len(), nnz),
        });
    }

    // Handle trivial cases
    if n_rows == 0 || n_cols == 0 {
        return Ok(MatchingResult {
            col_to_row: vec![NIL; n_cols],
            row_to_col: vec![NIL; n_rows],
            matching_size: 0,
            structural_rank: 0,
        });
    }

    // Initialize matching arrays
    // col_to_row[j] = row matched to column j (NIL if unmatched)
    // row_to_col[i] = column matched to row i (NIL if unmatched)
    let mut col_to_row: Vec<i32> = vec![NIL; n_cols];
    let mut row_to_col: Vec<i32> = vec![NIL; n_rows];

    // Distance array for BFS (distance from unmatched columns)
    // dist[j] = distance of column j from nearest unmatched column in augmenting path
    // dist[n_cols] = distance of NIL (sentinel for unmatched rows)
    let mut dist: Vec<i32> = vec![0; n_cols + 1];

    let mut matching_size = 0usize;

    // Main loop: BFS + DFS until no augmenting paths
    while bfs(
        n_rows,
        n_cols,
        col_ptrs,
        row_indices,
        &col_to_row,
        &row_to_col,
        &mut dist,
    ) {
        // Try to find augmenting paths from each unmatched column
        for j in 0..n_cols {
            if col_to_row[j] == NIL {
                if dfs(
                    j,
                    n_rows,
                    n_cols,
                    col_ptrs,
                    row_indices,
                    &mut col_to_row,
                    &mut row_to_col,
                    &mut dist,
                ) {
                    matching_size += 1;
                }
            }
        }
    }

    Ok(MatchingResult {
        col_to_row,
        row_to_col,
        matching_size,
        structural_rank: matching_size,
    })
}

/// BFS phase: Build level graph from unmatched columns.
///
/// Returns true if at least one augmenting path exists.
fn bfs(
    n_rows: usize,
    n_cols: usize,
    col_ptrs: &[i64],
    row_indices: &[i64],
    col_to_row: &[i32],
    row_to_col: &[i32],
    dist: &mut [i32],
) -> bool {
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(n_cols);

    // Initialize: unmatched columns have distance 0, others have INF
    for j in 0..n_cols {
        if col_to_row[j] == NIL {
            dist[j] = 0;
            queue.push_back(j);
        } else {
            dist[j] = INF;
        }
    }

    // Distance to NIL (unmatched rows) - this is what we're trying to reach
    dist[n_cols] = INF;

    // BFS traversal
    while let Some(j) = queue.pop_front() {
        // If we haven't reached NIL yet (still looking for augmenting paths)
        if dist[j] < dist[n_cols] {
            // Explore all rows adjacent to column j
            let start = col_ptrs[j] as usize;
            let end = col_ptrs[j + 1] as usize;

            for idx in start..end {
                let i = row_indices[idx] as usize;
                if i >= n_rows {
                    continue;
                }

                // Get the column matched to this row (or NIL if unmatched)
                let matched_col = row_to_col[i];

                // Determine the index for distance lookup
                let matched_idx = if matched_col == NIL {
                    n_cols // Use sentinel index for unmatched rows
                } else {
                    matched_col as usize
                };

                // If this matched column hasn't been visited
                if dist[matched_idx] == INF {
                    dist[matched_idx] = dist[j] + 1;
                    if matched_col != NIL {
                        queue.push_back(matched_col as usize);
                    }
                }
            }
        }
    }

    // Return true if we found a path to an unmatched row
    dist[n_cols] != INF
}

/// DFS phase: Find augmenting path from column j in the level graph.
///
/// Returns true if an augmenting path was found and the matching was augmented.
fn dfs(
    j: usize,
    n_rows: usize,
    n_cols: usize,
    col_ptrs: &[i64],
    row_indices: &[i64],
    col_to_row: &mut [i32],
    row_to_col: &mut [i32],
    dist: &mut [i32],
) -> bool {
    // Explore all rows adjacent to column j
    let start = col_ptrs[j] as usize;
    let end = col_ptrs[j + 1] as usize;

    for idx in start..end {
        let i = row_indices[idx] as usize;
        if i >= n_rows {
            continue;
        }

        // Get the column matched to this row
        let matched_col = row_to_col[i];

        // Determine the index for distance lookup
        let matched_idx = if matched_col == NIL {
            n_cols
        } else {
            matched_col as usize
        };

        // Follow only edges in the level graph (distance increases by 1)
        if dist[matched_idx] == dist[j] + 1 {
            // If row is unmatched (reached NIL) or can find augmenting path from matched column
            if matched_col == NIL
                || dfs(
                    matched_col as usize,
                    n_rows,
                    n_cols,
                    col_ptrs,
                    row_indices,
                    col_to_row,
                    row_to_col,
                    dist,
                )
            {
                // Augment: match column j with row i
                col_to_row[j] = i as i32;
                row_to_col[i] = j as i32;
                return true;
            }
        }
    }

    // No augmenting path found from this column
    // Mark as visited by setting distance to INF (prevents revisiting in this phase)
    dist[j] = INF;
    false
}

/// Compute maximum transversal (row permutation for diagonal nonzeros).
///
/// This is a convenience wrapper around hopcroft_karp that returns the result
/// in a format suitable for sparse matrix permutation.
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
/// Tuple of (col_to_row, row_to_col, structural_rank) where:
/// - `col_to_row[j]` = row index matched to column j (NIL/-1 if unmatched)
/// - `row_to_col[i]` = column index matched to row i (NIL/-1 if unmatched)
/// - `structural_rank` = size of maximum matching
pub fn maximum_transversal(
    n_rows: usize,
    n_cols: usize,
    col_ptrs: &[i64],
    row_indices: &[i64],
) -> Result<(Vec<i32>, Vec<i32>, usize)> {
    let result = hopcroft_karp(n_rows, n_cols, col_ptrs, row_indices)?;

    Ok((result.col_to_row, result.row_to_col, result.structural_rank))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hopcroft_karp_empty() {
        let col_ptrs = vec![0i64];
        let row_indices: Vec<i64> = vec![];

        let result = hopcroft_karp(0, 0, &col_ptrs, &row_indices).unwrap();

        assert_eq!(result.matching_size, 0);
        assert_eq!(result.structural_rank, 0);
    }

    #[test]
    fn test_hopcroft_karp_single_edge() {
        // 1x1 matrix with single nonzero
        let col_ptrs = vec![0i64, 1];
        let row_indices = vec![0i64];

        let result = hopcroft_karp(1, 1, &col_ptrs, &row_indices).unwrap();

        assert_eq!(result.matching_size, 1);
        assert_eq!(result.col_to_row[0], 0);
        assert_eq!(result.row_to_col[0], 0);
    }

    #[test]
    fn test_hopcroft_karp_diagonal() {
        // 3x3 diagonal matrix - perfect matching
        let col_ptrs = vec![0i64, 1, 2, 3];
        let row_indices = vec![0i64, 1, 2];

        let result = hopcroft_karp(3, 3, &col_ptrs, &row_indices).unwrap();

        assert_eq!(result.matching_size, 3);
        assert_eq!(result.structural_rank, 3);

        // Each column matched to corresponding row
        assert_eq!(result.col_to_row[0], 0);
        assert_eq!(result.col_to_row[1], 1);
        assert_eq!(result.col_to_row[2], 2);
    }

    #[test]
    fn test_hopcroft_karp_permutation_needed() {
        // Matrix where matching requires non-trivial assignment:
        // [. x]
        // [x x]
        // Column 0 can only match row 1
        // Column 1 can match row 0 or 1, but must match row 0
        let col_ptrs = vec![0i64, 1, 3];
        let row_indices = vec![1i64, 0, 1];

        let result = hopcroft_karp(2, 2, &col_ptrs, &row_indices).unwrap();

        assert_eq!(result.matching_size, 2);
        // Column 0 -> Row 1, Column 1 -> Row 0
        assert_eq!(result.col_to_row[0], 1);
        assert_eq!(result.col_to_row[1], 0);
    }

    #[test]
    fn test_hopcroft_karp_incomplete_matching() {
        // Matrix where perfect matching is impossible:
        // [x .]
        // [x .]
        // Both columns can only match row 0 or row 1
        // Actually this is:
        // Column 0: rows 0, 1
        // Column 1: empty
        let col_ptrs = vec![0i64, 2, 2];
        let row_indices = vec![0i64, 1];

        let result = hopcroft_karp(2, 2, &col_ptrs, &row_indices).unwrap();

        assert_eq!(result.matching_size, 1);
        assert!(result.col_to_row[0] == 0 || result.col_to_row[0] == 1);
        assert_eq!(result.col_to_row[1], NIL);
    }

    #[test]
    fn test_hopcroft_karp_rectangular_more_rows() {
        // 4x2 matrix
        // [x .]
        // [x x]
        // [. x]
        // [. x]
        let col_ptrs = vec![0i64, 2, 5];
        let row_indices = vec![0i64, 1, 1, 2, 3];

        let result = hopcroft_karp(4, 2, &col_ptrs, &row_indices).unwrap();

        assert_eq!(result.matching_size, 2);
        // Column 0 can match 0 or 1, Column 1 can match 1, 2, or 3
        // One valid matching: col0->row0, col1->row1
        // Or: col0->row1, col1->row2 (or row3)
    }

    #[test]
    fn test_hopcroft_karp_rectangular_more_cols() {
        // 2x4 matrix
        let col_ptrs = vec![0i64, 1, 2, 2, 1];
        // Column 0: row 0
        // Column 1: row 1
        // Column 2: empty
        // Column 3: row 0
        let col_ptrs = vec![0i64, 1, 2, 2, 3];
        let row_indices = vec![0i64, 1, 0];

        let result = hopcroft_karp(2, 4, &col_ptrs, &row_indices).unwrap();

        // Max matching is 2 (only 2 rows)
        assert_eq!(result.matching_size, 2);
    }

    #[test]
    fn test_hopcroft_karp_augmenting_path() {
        // Classic example requiring augmenting path:
        // [x x .]
        // [x . x]
        // [. x x]
        // Greedy might match: col0->row0, col1->row2, leaving col2 unmatched
        // But optimal is: col0->row1, col1->row0, col2->row2
        let col_ptrs = vec![0i64, 2, 4, 6];
        let row_indices = vec![0i64, 1, 0, 2, 1, 2];

        let result = hopcroft_karp(3, 3, &col_ptrs, &row_indices).unwrap();

        assert_eq!(result.matching_size, 3);
        assert_eq!(result.structural_rank, 3);

        // Verify it's a valid perfect matching
        let mut row_matched = vec![false; 3];
        for j in 0..3 {
            let i = result.col_to_row[j];
            assert!(i >= 0 && i < 3);
            assert!(!row_matched[i as usize], "Row {} matched twice", i);
            row_matched[i as usize] = true;
        }
    }

    #[test]
    fn test_maximum_transversal() {
        // Simple test of the convenience function
        let col_ptrs = vec![0i64, 1, 2, 3];
        let row_indices = vec![0i64, 1, 2];

        let (col_to_row, row_to_col, rank) =
            maximum_transversal(3, 3, &col_ptrs, &row_indices).unwrap();

        assert_eq!(rank, 3);
        assert_eq!(col_to_row.len(), 3);
        assert_eq!(row_to_col.len(), 3);

        // Verify bidirectional matching consistency
        for (j, &i) in col_to_row.iter().enumerate() {
            if i >= 0 {
                assert_eq!(
                    row_to_col[i as usize], j as i32,
                    "Bidirectional matching must hold: col {} -> row {}, but row {} -> col {}",
                    j, i, i, row_to_col[i as usize]
                );
            }
        }
    }

    #[test]
    fn test_structural_rank_singular() {
        // Structurally singular matrix:
        // [x x]
        // [x x]
        // Rank is 1 (only one independent row/column pattern)
        // Wait, structural rank is about matching, not linear algebra rank
        // Both columns have same row pattern, so max matching is 2
        let col_ptrs = vec![0i64, 2, 4];
        let row_indices = vec![0i64, 1, 0, 1];

        let result = hopcroft_karp(2, 2, &col_ptrs, &row_indices).unwrap();

        // Actually both columns can match different rows
        assert_eq!(result.matching_size, 2);
    }

    #[test]
    fn test_structural_rank_truly_singular() {
        // Truly structurally singular:
        // [x .]
        // [x .]
        // Both rows in column 0, nothing in column 1
        let col_ptrs = vec![0i64, 2, 2];
        let row_indices = vec![0i64, 1];

        let result = hopcroft_karp(2, 2, &col_ptrs, &row_indices).unwrap();

        assert_eq!(result.structural_rank, 1);
    }
}
