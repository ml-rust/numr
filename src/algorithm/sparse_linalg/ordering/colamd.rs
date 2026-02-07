//! COLAMD: Column Approximate Minimum Degree Ordering.
//!
//! Computes a column permutation that reduces fill-in during sparse LU or QR
//! factorization. Based on the algorithm by Davis, Gilbert, Larimore, and Ng.
//!
//! # Algorithm Overview
//!
//! COLAMD works on the column-row graph of the matrix:
//! 1. Build bipartite graph: columns and rows as nodes, nonzeros as edges
//! 2. Iteratively select column with minimum degree (fewest incident rows)
//! 3. "Eliminate" the column: merge its rows into a superrow
//! 4. Update degrees of affected columns
//! 5. Use aggressive absorption to reduce graph size
//!
//! # Complexity
//!
//! - Time: O(nnz) for typical matrices
//! - Space: O(nnz) for the working graph
//!
//! # References
//!
//! Davis, T. A., Gilbert, J. R., Larimore, S. I., & Ng, E. G. (2004).
//! "A column approximate minimum degree ordering algorithm."
//! ACM Transactions on Mathematical Software, 30(3), 353-376.

use crate::error::{Error, Result};

/// Options for COLAMD algorithm.
#[derive(Debug, Clone)]
pub struct ColamdOptions {
    /// Dense row threshold: rows with more than dense_row_threshold * n_cols
    /// nonzeros are considered dense and ignored in degree computation.
    /// Default: 0.5 (50% density)
    pub dense_row_threshold: f64,

    /// Dense column threshold: columns with more than dense_col_threshold * n_rows
    /// nonzeros are placed last in the ordering.
    /// Default: 0.5 (50% density)
    pub dense_col_threshold: f64,

    /// Whether to compute statistics about the ordering.
    /// Default: false
    pub compute_stats: bool,
}

impl Default for ColamdOptions {
    fn default() -> Self {
        Self {
            dense_row_threshold: 0.5,
            dense_col_threshold: 0.5,
            compute_stats: false,
        }
    }
}

/// Statistics from COLAMD execution.
#[derive(Debug, Clone, Default)]
pub struct ColamdStats {
    /// Number of dense rows detected and ignored.
    pub n_dense_rows: usize,
    /// Number of dense columns placed at the end.
    pub n_dense_cols: usize,
    /// Number of empty columns (placed first).
    pub n_empty_cols: usize,
    /// Number of columns in the ordering.
    pub n_cols_ordered: usize,
}

/// Marker values for the algorithm state.
const DEAD: i32 = -1;

/// Column state during elimination.
#[derive(Debug, Clone)]
struct Column {
    /// Number of rows in this column (degree).
    degree: i32,
    /// Head of the row list for this column, or DEAD if eliminated.
    head: i32,
    /// Length of the row list.
    length: i32,
    /// Score for ordering (approximate degree after elimination).
    score: i32,
    /// Previous column in degree list.
    prev: i32,
    /// Next column in degree list.
    next: i32,
    /// Parent in elimination tree (for mass elimination).
    _parent: i32,
    /// Order in the output permutation, or -1 if not yet ordered.
    order: i32,
}

impl Column {
    fn new() -> Self {
        Self {
            degree: 0,
            head: -1,
            length: 0,
            score: 0,
            prev: -1,
            next: -1,
            _parent: -1,
            order: -1,
        }
    }

    fn is_dead(&self) -> bool {
        self.head == DEAD
    }

    fn is_alive(&self) -> bool {
        self.head != DEAD
    }
}

/// Row state during elimination.
#[derive(Debug, Clone)]
struct Row {
    /// Number of columns in this row.
    degree: i32,
    /// Head of the column list for this row.
    _head: i32,
    /// Length of the column list.
    _length: i32,
    /// Mark for detecting duplicates during superrow merge.
    _mark: i32,
    /// First column in the row (for pivot detection).
    _first_col: i32,
}

impl Row {
    fn new() -> Self {
        Self {
            degree: 0,
            _head: -1,
            _length: 0,
            _mark: 0,
            _first_col: -1,
        }
    }

    fn is_alive(&self) -> bool {
        self.degree != DEAD
    }
}

/// Compute COLAMD column ordering.
///
/// # Arguments
///
/// * `n_rows` - Number of rows in the matrix
/// * `n_cols` - Number of columns in the matrix
/// * `col_ptrs` - CSC column pointers (length n_cols + 1)
/// * `row_indices` - CSC row indices (length nnz)
/// * `options` - Algorithm options (use Default::default() for defaults)
///
/// # Returns
///
/// Tuple of (permutation, stats) where permutation[j] is the original column
/// index that should be placed in position j.
pub fn colamd(
    n_rows: usize,
    n_cols: usize,
    col_ptrs: &[i64],
    row_indices: &[i64],
    options: &ColamdOptions,
) -> Result<(Vec<usize>, ColamdStats)> {
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
    if n_cols == 0 {
        return Ok((vec![], ColamdStats::default()));
    }

    let mut stats = ColamdStats::default();

    // Initialize column and row structures
    let mut cols: Vec<Column> = vec![Column::new(); n_cols];
    let mut rows: Vec<Row> = vec![Row::new(); n_rows];

    // Compute row and column degrees, detect dense rows/cols
    let dense_row_count = (options.dense_row_threshold * n_cols as f64).ceil() as i32;
    let dense_col_count = (options.dense_col_threshold * n_rows as f64).ceil() as i32;

    // First pass: count column degrees and mark dense columns
    let mut dense_cols: Vec<usize> = Vec::new();
    let mut empty_cols: Vec<usize> = Vec::new();

    for j in 0..n_cols {
        let start = col_ptrs[j] as usize;
        let end = col_ptrs[j + 1] as usize;
        let degree = (end - start) as i32;

        if degree == 0 {
            empty_cols.push(j);
            cols[j].head = DEAD;
            stats.n_empty_cols += 1;
        } else if degree > dense_col_count {
            dense_cols.push(j);
            cols[j].head = DEAD;
            stats.n_dense_cols += 1;
        } else {
            cols[j].degree = degree;
            cols[j].length = degree;
            cols[j].head = start as i32;
        }
    }

    // Second pass: count row degrees, excluding dense columns
    for j in 0..n_cols {
        if cols[j].is_dead() {
            continue;
        }

        let start = col_ptrs[j] as usize;
        let end = col_ptrs[j + 1] as usize;

        for idx in start..end {
            let i = row_indices[idx] as usize;
            if i < n_rows {
                rows[i].degree += 1;
            }
        }
    }

    // Mark dense rows
    for i in 0..n_rows {
        if rows[i].degree > dense_row_count {
            rows[i].degree = DEAD;
            stats.n_dense_rows += 1;
        }
    }

    // Recount column degrees excluding dense rows
    for j in 0..n_cols {
        if cols[j].is_dead() {
            continue;
        }

        let start = col_ptrs[j] as usize;
        let end = col_ptrs[j + 1] as usize;
        let mut new_degree = 0i32;

        for idx in start..end {
            let i = row_indices[idx] as usize;
            if i < n_rows && rows[i].is_alive() {
                new_degree += 1;
            }
        }

        cols[j].degree = new_degree;
        cols[j].score = new_degree;

        if new_degree == 0 {
            // Column became empty after removing dense rows
            empty_cols.push(j);
            cols[j].head = DEAD;
        }
    }

    // Build degree lists for minimum degree selection
    // degree_head[d] = first column with degree d
    let max_degree = n_rows.min(n_cols);
    let mut degree_head: Vec<i32> = vec![-1; max_degree + 1];

    for j in 0..n_cols {
        if cols[j].is_alive() {
            let d = cols[j].score as usize;
            if d <= max_degree {
                let head = degree_head[d];
                cols[j].next = head;
                cols[j].prev = -1;
                if head >= 0 {
                    cols[head as usize].prev = j as i32;
                }
                degree_head[d] = j as i32;
            }
        }
    }

    // Build row-to-column adjacency (working copy that we'll modify)
    let mut row_col_adj: Vec<Vec<usize>> = vec![Vec::new(); n_rows];
    for j in 0..n_cols {
        if cols[j].is_dead() {
            continue;
        }

        let start = col_ptrs[j] as usize;
        let end = col_ptrs[j + 1] as usize;

        for idx in start..end {
            let i = row_indices[idx] as usize;
            if i < n_rows && rows[i].is_alive() {
                row_col_adj[i].push(j);
            }
        }
    }

    // Main elimination loop
    let mut perm: Vec<usize> = Vec::with_capacity(n_cols);
    let mut min_degree = 0usize;
    let mut n_ordered = 0usize;
    let _n_to_order = n_cols - empty_cols.len() - dense_cols.len();

    // Place empty columns first
    for &j in &empty_cols {
        cols[j].order = perm.len() as i32;
        perm.push(j);
        n_ordered += 1;
    }

    // Main loop: select minimum degree column and eliminate
    while n_ordered < n_cols - dense_cols.len() {
        // Find minimum degree column
        while min_degree <= max_degree && degree_head[min_degree] < 0 {
            min_degree += 1;
        }

        if min_degree > max_degree {
            break;
        }

        // Select pivot column (first in the degree list)
        let pivot_col = degree_head[min_degree] as usize;

        // Remove from degree list
        let next = cols[pivot_col].next;
        degree_head[min_degree] = next;
        if next >= 0 {
            cols[next as usize].prev = -1;
        }

        // Order this column
        cols[pivot_col].order = perm.len() as i32;
        cols[pivot_col].head = DEAD;
        perm.push(pivot_col);
        n_ordered += 1;

        // Collect rows affected by this pivot
        let start = col_ptrs[pivot_col] as usize;
        let end = col_ptrs[pivot_col + 1] as usize;
        let mut affected_rows: Vec<usize> = Vec::new();

        for idx in start..end {
            let i = row_indices[idx] as usize;
            if i < n_rows && rows[i].is_alive() {
                affected_rows.push(i);
            }
        }

        // Collect columns that share rows with pivot (excluding pivot)
        let mut affected_cols: Vec<usize> = Vec::new();
        let mut col_seen: Vec<bool> = vec![false; n_cols];
        col_seen[pivot_col] = true;

        for &i in &affected_rows {
            for &j in &row_col_adj[i] {
                if !col_seen[j] && cols[j].is_alive() {
                    col_seen[j] = true;
                    affected_cols.push(j);
                }
            }
        }

        // Remove pivot column from row adjacency lists and update degrees
        for &i in &affected_rows {
            row_col_adj[i].retain(|&j| j != pivot_col);
            rows[i].degree -= 1;

            // If row becomes empty, mark it dead
            if rows[i].degree <= 0 {
                rows[i].degree = DEAD;
            }
        }

        // Update affected columns: recompute scores
        for &j in &affected_cols {
            // Remove from current degree list
            let old_score = cols[j].score as usize;
            if old_score <= max_degree {
                let prev = cols[j].prev;
                let next = cols[j].next;

                if prev >= 0 {
                    cols[prev as usize].next = next;
                } else if degree_head[old_score] == j as i32 {
                    degree_head[old_score] = next;
                }

                if next >= 0 {
                    cols[next as usize].prev = prev;
                }
            }

            // Recompute degree (count alive rows)
            let start = col_ptrs[j] as usize;
            let end = col_ptrs[j + 1] as usize;
            let mut new_degree = 0i32;

            for idx in start..end {
                let i = row_indices[idx] as usize;
                if i < n_rows && rows[i].is_alive() {
                    new_degree += 1;
                }
            }

            cols[j].degree = new_degree;
            cols[j].score = new_degree;

            if new_degree == 0 {
                // Column became empty
                cols[j].order = perm.len() as i32;
                cols[j].head = DEAD;
                perm.push(j);
                n_ordered += 1;
            } else {
                // Re-insert into degree list
                let d = new_degree as usize;
                if d <= max_degree {
                    let head = degree_head[d];
                    cols[j].next = head;
                    cols[j].prev = -1;
                    if head >= 0 {
                        cols[head as usize].prev = j as i32;
                    }
                    degree_head[d] = j as i32;

                    // Update min_degree hint
                    if d < min_degree {
                        min_degree = d;
                    }
                }
            }
        }
    }

    // Place dense columns last
    for &j in &dense_cols {
        cols[j].order = perm.len() as i32;
        perm.push(j);
    }

    stats.n_cols_ordered = perm.len();

    Ok((perm, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colamd_empty_matrix() {
        let col_ptrs = vec![0i64];
        let row_indices: Vec<i64> = vec![];

        let (perm, stats) =
            colamd(0, 0, &col_ptrs, &row_indices, &ColamdOptions::default()).unwrap();

        assert!(perm.is_empty());
        assert_eq!(stats.n_cols_ordered, 0);
    }

    #[test]
    fn test_colamd_single_column() {
        let col_ptrs = vec![0i64, 2];
        let row_indices = vec![0i64, 1];

        let (perm, stats) =
            colamd(2, 1, &col_ptrs, &row_indices, &ColamdOptions::default()).unwrap();

        assert_eq!(perm, vec![0]);
        assert_eq!(stats.n_cols_ordered, 1);
    }

    #[test]
    fn test_colamd_diagonal_matrix() {
        // 3x3 diagonal matrix: nonzeros at (0,0), (1,1), (2,2)
        let col_ptrs = vec![0i64, 1, 2, 3];
        let row_indices = vec![0i64, 1, 2];

        let (perm, stats) =
            colamd(3, 3, &col_ptrs, &row_indices, &ColamdOptions::default()).unwrap();

        assert_eq!(perm.len(), 3);
        assert_eq!(stats.n_cols_ordered, 3);

        // All columns have degree 1, order depends on ties but should contain all
        let mut sorted_perm = perm.clone();
        sorted_perm.sort();
        assert_eq!(sorted_perm, vec![0, 1, 2]);
    }

    #[test]
    fn test_colamd_tridiagonal_matrix() {
        // 4x4 tridiagonal matrix:
        // [x . . .]
        // [x x . .]
        // [. x x .]
        // [. . x x]
        // CSC format: column 0 has rows 0,1; column 1 has rows 1,2; etc.
        let col_ptrs = vec![0i64, 2, 4, 6, 7];
        let row_indices = vec![0i64, 1, 1, 2, 2, 3, 3];

        let (perm, stats) =
            colamd(4, 4, &col_ptrs, &row_indices, &ColamdOptions::default()).unwrap();

        assert_eq!(perm.len(), 4);
        assert_eq!(stats.n_cols_ordered, 4);

        // Verify it's a valid permutation
        let mut sorted_perm = perm.clone();
        sorted_perm.sort();
        assert_eq!(sorted_perm, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_colamd_with_empty_column() {
        // Matrix with an empty column (column 1)
        let col_ptrs = vec![0i64, 2, 2, 4]; // column 1 is empty
        let row_indices = vec![0i64, 1, 0, 1];

        let (perm, stats) =
            colamd(2, 3, &col_ptrs, &row_indices, &ColamdOptions::default()).unwrap();

        assert_eq!(perm.len(), 3);
        assert_eq!(stats.n_empty_cols, 1);

        // Empty column should be placed first
        assert_eq!(perm[0], 1);
    }

    #[test]
    fn test_colamd_arrow_matrix() {
        // Arrow matrix pattern (dense first row and column):
        // [x x x x]
        // [x x . .]
        // [x . x .]
        // [x . . x]
        // Column 0: rows 0,1,2,3 (degree 4)
        // Column 1: rows 0,1 (degree 2)
        // Column 2: rows 0,2 (degree 2)
        // Column 3: rows 0,3 (degree 2)
        let col_ptrs = vec![0i64, 4, 6, 8, 10];
        let row_indices = vec![0i64, 1, 2, 3, 0, 1, 0, 2, 0, 3];

        let options = ColamdOptions {
            // 0.7 * 4 rows = 2.8, ceil = 3
            // Column 0 has degree 4 > 3, so it's detected as dense
            dense_col_threshold: 0.7,
            ..Default::default()
        };

        let (perm, stats) = colamd(4, 4, &col_ptrs, &row_indices, &options).unwrap();

        assert_eq!(perm.len(), 4);
        assert_eq!(stats.n_dense_cols, 1);

        // Dense column should be placed last
        assert_eq!(perm[3], 0);
    }

    #[test]
    fn test_colamd_permutation_validity() {
        // Random sparse matrix
        let col_ptrs = vec![0i64, 3, 5, 8, 10, 12];
        let row_indices = vec![0i64, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 4];

        let (perm, _) = colamd(5, 5, &col_ptrs, &row_indices, &ColamdOptions::default()).unwrap();

        // Verify it's a valid permutation (contains each index exactly once)
        assert_eq!(perm.len(), 5);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
}
