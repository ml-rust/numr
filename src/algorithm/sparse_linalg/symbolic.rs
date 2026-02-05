//! Shared symbolic factorization implementations.
//!
//! These functions perform pure CPU-based graph analysis on sparse matrix
//! structure. They are used by all backends (CPU, CUDA, WebGPU) since symbolic
//! analysis is inherently sequential and requires HashMap-based lookups.
//!
//! # Design Rationale
//!
//! Symbolic factorization analyzes sparsity patterns to:
//! - Determine fill-in structure for ILU(k)
//! - Precompute update schedules for efficient numeric factorization
//! - Build dependency graphs for level scheduling
//!
//! This is done once per matrix structure and reused for multiple numeric
//! factorizations (e.g., Newton iterations where structure stays constant).

use std::collections::HashMap;

use super::types::{IlukSymbolic, SymbolicIlu0};
use crate::algorithm::sparse_linalg::IluFillLevel;
use crate::error::Result;

/// Compute ILU(0) symbolic factorization from raw CSR data.
///
/// Analyzes the sparsity pattern to create an efficient update schedule
/// for numeric factorization. This avoids hash map lookups during the
/// numeric phase.
///
/// # Arguments
///
/// * `n` - Matrix dimension (n x n)
/// * `row_ptrs` - CSR row pointers
/// * `col_indices` - CSR column indices
///
/// # Returns
///
/// Symbolic factorization data including L/U patterns and update schedule.
pub fn ilu0_symbolic_impl(n: usize, row_ptrs: &[i64], col_indices: &[i64]) -> Result<SymbolicIlu0> {
    // Build column-to-index map for fast lookup
    let mut col_to_idx: Vec<HashMap<usize, usize>> = vec![HashMap::new(); n];
    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;
        for idx in start..end {
            let j = col_indices[idx] as usize;
            col_to_idx[i].insert(j, idx);
        }
    }

    // Build L and U patterns
    let mut l_row_ptrs = vec![0i64; n + 1];
    let mut l_col_indices = Vec::new();
    let mut u_row_ptrs = vec![0i64; n + 1];
    let mut u_col_indices = Vec::new();
    let mut diag_positions = vec![0usize; n];

    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;

        for idx in start..end {
            let j = col_indices[idx] as usize;
            if j < i {
                l_col_indices.push(j as i64);
            } else {
                if j == i {
                    diag_positions[i] = u_col_indices.len();
                }
                u_col_indices.push(j as i64);
            }
        }

        l_row_ptrs[i + 1] = l_col_indices.len() as i64;
        u_row_ptrs[i + 1] = u_col_indices.len() as i64;
    }

    // Precompute update schedule for IKJ factorization
    // For each row i, we need to process columns k < i and update A[i,j] -= L[i,k]*U[k,j]
    let mut update_schedule = Vec::with_capacity(n);

    for i in 0..n {
        let mut row_updates = Vec::new();
        let start_i = row_ptrs[i] as usize;
        let end_i = row_ptrs[i + 1] as usize;

        // For each k < i where A[i,k] exists (will become L[i,k])
        for idx_ik in start_i..end_i {
            let k = col_indices[idx_ik] as usize;
            if k >= i {
                break;
            }

            // Build list of updates: (j, idx_ij, idx_kj)
            let mut updates_for_k = Vec::new();
            let start_k = row_ptrs[k] as usize;
            let end_k = row_ptrs[k + 1] as usize;

            for idx_kj in start_k..end_k {
                let j = col_indices[idx_kj] as usize;
                if j <= k {
                    continue;
                }

                // Check if A[i,j] exists
                if let Some(&idx_ij) = col_to_idx[i].get(&j) {
                    updates_for_k.push((j, idx_ij, idx_kj));
                }
            }

            if !updates_for_k.is_empty() || col_to_idx[k].contains_key(&k) {
                row_updates.push((k, idx_ik, updates_for_k));
            }
        }

        update_schedule.push(row_updates);
    }

    Ok(SymbolicIlu0 {
        n,
        l_row_ptrs,
        l_col_indices,
        u_row_ptrs,
        u_col_indices,
        diag_positions,
        update_schedule,
    })
}

/// Compute ILU(k) symbolic factorization from raw CSR data.
///
/// Uses level-of-fill algorithm to determine which fill-in entries to keep.
/// An entry (i,j) has level = min over all k of (level[i,k] + level[k,j] + 1).
/// Original entries have level 0, and we keep entries with level <= fill_level.
///
/// # Arguments
///
/// * `n` - Matrix dimension (n x n)
/// * `row_ptrs` - CSR row pointers
/// * `col_indices` - CSR column indices
/// * `fill_level` - Maximum fill level to keep
///
/// # Returns
///
/// Symbolic factorization data including L/U patterns with fill-in.
pub fn iluk_symbolic_impl(
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
    fill_level: IluFillLevel,
) -> Result<IlukSymbolic> {
    let max_level = fill_level.level();

    // Build initial level-of-fill map from original matrix entries
    // level_of_fill[i][j] = level of entry (i,j), None if entry doesn't exist
    let mut level_of_fill: Vec<HashMap<usize, usize>> = vec![HashMap::new(); n];

    // Initialize with original entries (level 0)
    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;
        for idx in start..end {
            let j = col_indices[idx] as usize;
            level_of_fill[i].insert(j, 0);
        }
    }

    // Symbolic factorization with level-of-fill tracking
    // For each row i, for each k < i where (i,k) exists:
    //   For each j > k where (k,j) exists:
    //     new_level = level[i,k] + level[k,j] + 1
    //     if new_level <= max_level:
    //       level[i,j] = min(level[i,j], new_level)
    for i in 0..n {
        // Get columns k < i that exist in row i
        let cols_below_diag: Vec<usize> = level_of_fill[i]
            .keys()
            .filter(|&&k| k < i)
            .copied()
            .collect();

        for k in cols_below_diag {
            let level_ik = level_of_fill[i][&k];

            // Get ALL columns j that exist in row k (for level-of-fill computation)
            // Note: This differs from numeric IKJ which only considers j > k
            let cols_in_row_k: Vec<(usize, usize)> =
                level_of_fill[k].iter().map(|(&j, &lev)| (j, lev)).collect();

            for (j, level_kj) in cols_in_row_k {
                // Skip j == k (diagonal) and j < k when computing L part updates
                // For level-of-fill, we need j != k (diagonal doesn't get fill)
                if j == k {
                    continue;
                }
                let new_level = level_ik + level_kj + 1;
                if new_level <= max_level {
                    let entry = level_of_fill[i].entry(j).or_insert(new_level);
                    if new_level < *entry {
                        *entry = new_level;
                    }
                }
            }
        }
    }

    // Build L and U patterns from level_of_fill
    let mut l_row_ptrs = vec![0i64; n + 1];
    let mut l_col_indices = Vec::new();
    let mut u_row_ptrs = vec![0i64; n + 1];
    let mut u_col_indices = Vec::new();

    for i in 0..n {
        // Get sorted column indices for this row
        let mut cols: Vec<usize> = level_of_fill[i].keys().copied().collect();
        cols.sort_unstable();

        for j in cols {
            if j < i {
                l_col_indices.push(j as i64);
            } else {
                u_col_indices.push(j as i64);
            }
        }

        l_row_ptrs[i + 1] = l_col_indices.len() as i64;
        u_row_ptrs[i + 1] = u_col_indices.len() as i64;
    }

    Ok(IlukSymbolic {
        n,
        level_of_fill,
        row_ptrs_l: l_row_ptrs,
        col_indices_l: l_col_indices,
        row_ptrs_u: u_row_ptrs,
        col_indices_u: u_col_indices,
        fill_level,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ilu0_symbolic_basic() {
        // Simple 3x3 tridiagonal matrix structure
        let row_ptrs = vec![0i64, 2, 5, 7];
        let col_indices = vec![0i64, 1, 0, 1, 2, 1, 2];

        let symbolic = ilu0_symbolic_impl(3, &row_ptrs, &col_indices).unwrap();

        assert_eq!(symbolic.n, 3);
        assert_eq!(symbolic.l_row_ptrs.len(), 4);
        assert_eq!(symbolic.u_row_ptrs.len(), 4);
    }

    #[test]
    fn test_iluk_symbolic_level0() {
        // Level 0 should preserve original structure
        let row_ptrs = vec![0i64, 2, 5, 7];
        let col_indices = vec![0i64, 1, 0, 1, 2, 1, 2];

        let symbolic = iluk_symbolic_impl(3, &row_ptrs, &col_indices, IluFillLevel::Zero).unwrap();

        assert_eq!(symbolic.n, 3);
        // L should have 2 entries (below diagonal)
        assert_eq!(symbolic.col_indices_l.len(), 2);
        // U should have 5 entries (diagonal and above)
        assert_eq!(symbolic.col_indices_u.len(), 5);
    }

    #[test]
    fn test_iluk_symbolic_level1_fill() {
        // Level 1 should add fill-in
        let row_ptrs = vec![0i64, 2, 5, 7];
        let col_indices = vec![0i64, 1, 0, 1, 2, 1, 2];

        let symbolic = iluk_symbolic_impl(3, &row_ptrs, &col_indices, IluFillLevel::One).unwrap();

        // With level 1 fill, row 2 should have entry (2,0) since:
        // level[2,1] = 0, level[1,0] = 0 → level[2,0] = 0+0+1 = 1
        assert!(symbolic.level_of_fill[2].contains_key(&0));
    }
}
