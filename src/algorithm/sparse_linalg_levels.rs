//! Level scheduling for sparse linear algebra
//!
//! This module provides algorithms to compute dependency levels for sparse matrices,
//! enabling parallel execution of inherently sequential algorithms like ILU, IC,
//! and triangular solve.
//!
//! # Background
//!
//! Sparse triangular operations have row-to-row dependencies:
//! - Row i depends on all rows j < i where A[i,j] ≠ 0
//!
//! Level scheduling groups rows into "levels" where:
//! - All rows within a level are independent (can execute in parallel)
//! - Levels must execute sequentially
//!
//! # Algorithm
//!
//! For a lower triangular matrix:
//! ```text
//! level[i] = max(level[j] for all j where L[i,j] ≠ 0 and j < i) + 1
//! level[0] = 0  (first row has no dependencies)
//! ```
//!
//! For an upper triangular matrix (backward):
//! ```text
//! level[i] = max(level[j] for all j where U[i,j] ≠ 0 and j > i) + 1
//! level[n-1] = 0  (last row has no dependencies in backward solve)
//! ```

use crate::error::Result;

/// Result of level analysis for a sparse triangular matrix.
///
/// Contains the level assignment for each row and organized level structure
/// for efficient parallel dispatch.
#[derive(Debug, Clone)]
pub struct LevelSchedule {
    /// Level assignment for each row: level_of_row[i] = level of row i
    pub level_of_row: Vec<usize>,

    /// Rows grouped by level: rows_per_level[l] = Vec of row indices at level l
    pub rows_per_level: Vec<Vec<usize>>,

    /// Total number of levels (depth of the dependency DAG)
    pub num_levels: usize,

    /// Maximum parallelism (largest level size)
    pub max_parallelism: usize,
}

/// Compute level schedule for a lower triangular sparse matrix (CSR format).
///
/// # Arguments
///
/// * `n` - Matrix dimension (n x n)
/// * `row_ptrs` - CSR row pointers [n+1]
/// * `col_indices` - CSR column indices [nnz]
///
/// # Returns
///
/// Level schedule with row assignments and grouped levels.
pub fn compute_levels_lower(
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
) -> Result<LevelSchedule> {
    let mut level_of_row = vec![0usize; n];

    // Forward pass: compute level of each row
    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;

        let mut max_dep_level = 0usize;

        // Find maximum level among dependencies (columns j < i)
        for idx in start..end {
            let j = col_indices[idx] as usize;
            if j < i {
                max_dep_level = max_dep_level.max(level_of_row[j] + 1);
            }
        }

        level_of_row[i] = max_dep_level;
    }

    // Group rows by level
    let num_levels = level_of_row.iter().max().map(|&x| x + 1).unwrap_or(0);
    let mut rows_per_level: Vec<Vec<usize>> = vec![Vec::new(); num_levels];

    for (row, &level) in level_of_row.iter().enumerate() {
        rows_per_level[level].push(row);
    }

    let max_parallelism = rows_per_level.iter().map(|v| v.len()).max().unwrap_or(0);

    Ok(LevelSchedule {
        level_of_row,
        rows_per_level,
        num_levels,
        max_parallelism,
    })
}

/// Compute level schedule for an upper triangular sparse matrix (CSR format).
///
/// For backward substitution, dependencies flow from higher to lower indices.
///
/// # Arguments
///
/// * `n` - Matrix dimension (n x n)
/// * `row_ptrs` - CSR row pointers [n+1]
/// * `col_indices` - CSR column indices [nnz]
///
/// # Returns
///
/// Level schedule with row assignments and grouped levels.
pub fn compute_levels_upper(
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
) -> Result<LevelSchedule> {
    let mut level_of_row = vec![0usize; n];

    // Backward pass: compute level of each row (starting from last row)
    for i in (0..n).rev() {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;

        let mut max_dep_level = 0usize;

        // Find maximum level among dependencies (columns j > i)
        for idx in start..end {
            let j = col_indices[idx] as usize;
            if j > i {
                max_dep_level = max_dep_level.max(level_of_row[j] + 1);
            }
        }

        level_of_row[i] = max_dep_level;
    }

    // Group rows by level
    let num_levels = level_of_row.iter().max().map(|&x| x + 1).unwrap_or(0);
    let mut rows_per_level: Vec<Vec<usize>> = vec![Vec::new(); num_levels];

    for (row, &level) in level_of_row.iter().enumerate() {
        rows_per_level[level].push(row);
    }

    let max_parallelism = rows_per_level.iter().map(|v| v.len()).max().unwrap_or(0);

    Ok(LevelSchedule {
        level_of_row,
        rows_per_level,
        num_levels,
        max_parallelism,
    })
}

/// Compute level schedule for ILU(0) factorization.
///
/// ILU(0) has dependencies based on the full matrix structure (not just lower triangle).
/// Row i depends on all rows k < i where A[i,k] ≠ 0 (to compute L[i,k] = A[i,k] / U[k,k]).
///
/// # Arguments
///
/// * `n` - Matrix dimension (n x n)
/// * `row_ptrs` - CSR row pointers [n+1]
/// * `col_indices` - CSR column indices [nnz]
///
/// # Returns
///
/// Level schedule for ILU factorization.
pub fn compute_levels_ilu(
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
) -> Result<LevelSchedule> {
    // For ILU, dependencies are same as lower triangular solve on the original matrix
    compute_levels_lower(n, row_ptrs, col_indices)
}

/// Flatten level schedule into arrays suitable for GPU execution.
///
/// Returns:
/// - `level_ptrs`: Start index of each level in `level_rows` [num_levels + 1]
/// - `level_rows`: Row indices sorted by level [n]
pub fn flatten_levels(schedule: &LevelSchedule) -> (Vec<i32>, Vec<i32>) {
    let n: usize = schedule.level_of_row.len();
    let mut level_ptrs = Vec::with_capacity(schedule.num_levels + 1);
    let mut level_rows = Vec::with_capacity(n);

    level_ptrs.push(0i32);

    for level_row_list in &schedule.rows_per_level {
        for &row in level_row_list {
            level_rows.push(row as i32);
        }
        level_ptrs.push(level_rows.len() as i32);
    }

    (level_ptrs, level_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levels_lower_diagonal() {
        // Diagonal matrix: no dependencies, all rows at level 0
        let row_ptrs = vec![0i64, 1, 2, 3];
        let col_indices = vec![0i64, 1, 2];

        let schedule = compute_levels_lower(3, &row_ptrs, &col_indices).unwrap();

        assert_eq!(schedule.num_levels, 1);
        assert_eq!(schedule.level_of_row, vec![0, 0, 0]);
        assert_eq!(schedule.max_parallelism, 3);
    }

    #[test]
    fn test_levels_lower_tridiagonal() {
        // Tridiagonal lower:
        // [x . .]
        // [x x .]
        // [. x x]
        // Row 0: level 0 (no deps)
        // Row 1: depends on row 0 -> level 1
        // Row 2: depends on row 1 -> level 2
        let row_ptrs = vec![0i64, 1, 3, 5];
        let col_indices = vec![0i64, 0, 1, 1, 2];

        let schedule = compute_levels_lower(3, &row_ptrs, &col_indices).unwrap();

        assert_eq!(schedule.num_levels, 3);
        assert_eq!(schedule.level_of_row, vec![0, 1, 2]);
        assert_eq!(schedule.max_parallelism, 1); // No parallelism in tridiagonal
    }

    #[test]
    fn test_levels_lower_with_parallelism() {
        // Matrix with parallelism:
        // [x . . .]
        // [. x . .]
        // [x . x .]
        // [. x . x]
        // Row 0, 1: level 0 (no deps on previous rows)
        // Row 2: depends on row 0 -> level 1
        // Row 3: depends on row 1 -> level 1
        let row_ptrs = vec![0i64, 1, 2, 4, 6];
        let col_indices = vec![0i64, 1, 0, 2, 1, 3];

        let schedule = compute_levels_lower(4, &row_ptrs, &col_indices).unwrap();

        assert_eq!(schedule.num_levels, 2);
        assert_eq!(schedule.level_of_row, vec![0, 0, 1, 1]);
        assert_eq!(schedule.max_parallelism, 2);
        assert_eq!(schedule.rows_per_level[0], vec![0, 1]);
        assert_eq!(schedule.rows_per_level[1], vec![2, 3]);
    }

    #[test]
    fn test_levels_upper() {
        // Upper triangular:
        // [x x .]
        // [. x x]
        // [. . x]
        // Backward: Row 2 level 0, Row 1 depends on 2 -> level 1, Row 0 depends on 1 -> level 2
        let row_ptrs = vec![0i64, 2, 4, 5];
        let col_indices = vec![0i64, 1, 1, 2, 2];

        let schedule = compute_levels_upper(3, &row_ptrs, &col_indices).unwrap();

        assert_eq!(schedule.num_levels, 3);
        assert_eq!(schedule.level_of_row, vec![2, 1, 0]);
    }

    #[test]
    fn test_flatten_levels() {
        let schedule = LevelSchedule {
            level_of_row: vec![0, 0, 1, 1],
            rows_per_level: vec![vec![0, 1], vec![2, 3]],
            num_levels: 2,
            max_parallelism: 2,
        };

        let (level_ptrs, level_rows) = flatten_levels(&schedule);

        assert_eq!(level_ptrs, vec![0, 2, 4]);
        assert_eq!(level_rows, vec![0, 1, 2, 3]);
    }
}
