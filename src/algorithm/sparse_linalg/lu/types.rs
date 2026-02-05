//! Types for sparse LU factorization
//!
//! Contains factorization results, symbolic structures, and options.

use crate::runtime::Runtime;
use crate::sparse::CscData;

// ============================================================================
// LU Factorization Types
// ============================================================================

/// Result of sparse LU factorization: PA = LU
///
/// Contains the L and U factors in CSC format, plus row permutation.
/// - L is unit lower triangular (ones on diagonal, stored implicitly)
/// - U is upper triangular (diagonal stored explicitly)
/// - P is the row permutation from partial pivoting
#[derive(Debug, Clone)]
pub struct LuFactors<R: Runtime> {
    /// Lower triangular factor (unit diagonal, stored implicitly as zeros)
    /// Shape: [n, n] in CSC format
    pub l: CscData<R>,

    /// Upper triangular factor (includes diagonal)
    /// Shape: [n, n] in CSC format
    pub u: CscData<R>,

    /// Row permutation from partial pivoting
    /// perm[k] = row that was pivoted to position k
    pub row_perm: Vec<usize>,

    /// Inverse row permutation
    /// inv_perm[row] = position of original row
    pub row_perm_inv: Vec<usize>,
}

/// Symbolic factorization for sparse LU
///
/// Precomputed structure that describes the sparsity pattern of L and U
/// factors. This is computed by solvr's symbolic analysis and reused
/// for multiple numeric factorizations with the same pattern.
///
/// # Components
///
/// - **Elimination tree**: Parent pointers for the column dependency graph
/// - **Column reach**: For each column k, which prior columns affect it
/// - **Post-order**: Topological ordering of columns for efficient traversal
/// - **Pattern**: Predicted nonzero positions in L and U
#[derive(Debug, Clone)]
pub struct LuSymbolic {
    /// Matrix dimension
    pub n: usize,

    /// Elimination tree: etree[j] = parent of column j, or n if root
    pub etree: Vec<usize>,

    /// Post-order traversal of elimination tree
    /// Columns should be processed in this order for efficiency
    pub post_order: Vec<usize>,

    /// For each column k, the list of columns j < k that contribute to it
    /// reach[k] contains column indices in topological order
    pub reach: Vec<Vec<usize>>,

    /// Column pointers for L pattern (predicted fill-in positions)
    pub l_col_ptrs: Vec<i64>,

    /// Row indices for L pattern
    pub l_row_indices: Vec<i64>,

    /// Column pointers for U pattern
    pub u_col_ptrs: Vec<i64>,

    /// Row indices for U pattern
    pub u_row_indices: Vec<i64>,

    /// Workspace sizes needed for numeric factorization
    pub workspace_size: usize,
}

impl LuSymbolic {
    /// Create a minimal symbolic structure for a matrix with no fill-in
    /// (useful for testing or when symbolic analysis is skipped)
    pub fn identity(n: usize) -> Self {
        // For identity-like structure: L and U are both diagonal
        let col_ptrs: Vec<i64> = (0..=n).map(|i| i as i64).collect();
        let row_indices: Vec<i64> = (0..n).map(|i| i as i64).collect();

        Self {
            n,
            etree: (1..=n).collect(), // Each column's parent is the next column
            post_order: (0..n).collect(),
            reach: vec![vec![]; n],
            l_col_ptrs: col_ptrs.clone(),
            l_row_indices: row_indices.clone(),
            u_col_ptrs: col_ptrs,
            u_row_indices: row_indices,
            workspace_size: n,
        }
    }

    /// Estimated number of nonzeros in L
    pub fn l_nnz(&self) -> usize {
        self.l_row_indices.len()
    }

    /// Estimated number of nonzeros in U
    pub fn u_nnz(&self) -> usize {
        self.u_row_indices.len()
    }
}

/// Configuration for sparse LU factorization
#[derive(Debug, Clone)]
pub struct LuOptions {
    /// Pivot tolerance for partial pivoting (default: 1.0 = full pivoting)
    /// Values in [0, 1] where 0 = no pivoting, 1 = always choose largest
    pub pivot_tolerance: f64,

    /// Minimum acceptable pivot magnitude (default: 1e-12)
    /// Pivots smaller than this cause an error or diagonal shift
    pub pivot_threshold: f64,

    /// Diagonal shift to add when pivot is too small (default: 0.0)
    /// If > 0, adds this value instead of failing on small pivots
    pub diagonal_shift: f64,

    /// Whether to check for exact zeros (default: true)
    /// If false, very small values may be treated as zero
    pub check_zeros: bool,
}

impl Default for LuOptions {
    fn default() -> Self {
        Self {
            pivot_tolerance: 1.0,
            pivot_threshold: 1e-12,
            diagonal_shift: 0.0,
            check_zeros: true,
        }
    }
}

impl LuOptions {
    /// Create options for no pivoting (useful for symmetric positive definite)
    pub fn no_pivoting() -> Self {
        Self {
            pivot_tolerance: 0.0,
            ..Default::default()
        }
    }

    /// Create options with diagonal shift for numerical stability
    pub fn with_diagonal_shift(shift: f64) -> Self {
        Self {
            diagonal_shift: shift,
            ..Default::default()
        }
    }
}

/// Metrics from LU factorization for diagnostics
#[derive(Debug, Clone)]
pub struct LuMetrics {
    /// Number of non-zeros in original matrix
    pub original_nnz: usize,

    /// Number of non-zeros in L factor
    pub l_nnz: usize,

    /// Number of non-zeros in U factor
    pub u_nnz: usize,

    /// Fill ratio: (l_nnz + u_nnz) / original_nnz
    pub fill_ratio: f64,

    /// Number of pivots below threshold that required shifting
    pub small_pivots: usize,

    /// Number of row swaps performed during pivoting
    pub row_swaps: usize,

    /// Maximum pivot growth factor: max(|U[i,i]|) / max(|A[i,j]|)
    pub pivot_growth: f64,
}

// ============================================================================
// Simple Symbolic Structure (for use without solvr)
// ============================================================================

/// Simplified symbolic structure for when full symbolic analysis isn't available
///
/// This uses the original matrix pattern as a starting point and allows
/// the numeric factorization to discover fill-in dynamically.
#[derive(Debug, Clone)]
pub struct LuSymbolicSimple {
    /// Matrix dimension
    pub n: usize,

    /// Column pointers from original matrix
    pub col_ptrs: Vec<i64>,

    /// Row indices from original matrix
    pub row_indices: Vec<i64>,

    /// For each column, position of diagonal element (if any)
    pub diag_positions: Vec<Option<usize>>,
}

impl LuSymbolicSimple {
    /// Create from CSC matrix pattern
    pub fn from_pattern(col_ptrs: &[i64], row_indices: &[i64], n: usize) -> Self {
        let mut diag_positions = vec![None; n];

        for col in 0..n {
            let start = col_ptrs[col] as usize;
            let end = col_ptrs[col + 1] as usize;

            for idx in start..end {
                let row = row_indices[idx] as usize;
                if row == col {
                    diag_positions[col] = Some(idx);
                    break;
                }
            }
        }

        Self {
            n,
            col_ptrs: col_ptrs.to_vec(),
            row_indices: row_indices.to_vec(),
            diag_positions,
        }
    }
}
