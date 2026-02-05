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

// ============================================================================
// Workspace for Allocation-Free Refactorization
// ============================================================================

/// Pre-allocated workspace buffers for sparse LU numeric factorization.
///
/// This struct enables allocation-free numeric refactorization when the sparsity
/// pattern remains unchanged (common in Newton iterations for ODEs/DAEs).
///
/// # Usage
///
/// ```ignore
/// // First factorization: allocate workspace
/// let symbolic = compute_symbolic(&a);
/// let mut workspace = LuWorkspace::new(symbolic.n, &symbolic);
/// let factors = sparse_lu_cpu_with_workspace(&a, &symbolic, &options, &mut workspace)?;
///
/// // Subsequent factorizations: reuse workspace (no allocations!)
/// let factors2 = sparse_lu_cpu_with_workspace(&a2, &symbolic, &options, &mut workspace)?;
/// ```
///
/// # Thread Safety
///
/// `LuWorkspace` is NOT `Send` or `Sync`. Each thread should have its own workspace.
#[derive(Debug, Clone)]
pub struct LuWorkspace {
    /// Dense work vector for scatter/gather operations [n]
    pub(crate) work: Vec<f64>,

    /// Scratch space for row permutation [n]
    pub(crate) row_perm: Vec<usize>,

    /// Scratch space for inverse row permutation [n]
    pub(crate) row_perm_inv: Vec<usize>,

    /// Visited flags for DFS in simple factorization [n]
    pub(crate) visited: Vec<bool>,

    /// Stack for reach computation in simple factorization
    pub(crate) reach_stack: Vec<usize>,

    /// Pre-allocated L values buffer
    pub(crate) l_values_buffer: Vec<f64>,

    /// Pre-allocated L row indices buffer
    pub(crate) l_row_indices_buffer: Vec<i64>,

    /// Pre-allocated U values buffer
    pub(crate) u_values_buffer: Vec<f64>,

    /// Pre-allocated U row indices buffer
    pub(crate) u_row_indices_buffer: Vec<i64>,

    /// Matrix dimension this workspace was created for
    pub(crate) n: usize,
}

impl LuWorkspace {
    /// Create a new workspace for matrices of dimension `n`.
    ///
    /// # Arguments
    ///
    /// * `n` - Matrix dimension
    /// * `symbolic` - Symbolic factorization (used to estimate buffer sizes)
    pub fn new(n: usize, symbolic: &LuSymbolic) -> Self {
        // Estimate L and U sizes from symbolic analysis (with some slack)
        let l_capacity = (symbolic.l_nnz() as f64 * 1.2) as usize;
        let u_capacity = (symbolic.u_nnz() as f64 * 1.2) as usize;

        Self {
            work: vec![0.0; n],
            row_perm: (0..n).collect(),
            row_perm_inv: (0..n).collect(),
            visited: vec![false; n],
            reach_stack: Vec::with_capacity(n),
            l_values_buffer: Vec::with_capacity(l_capacity),
            l_row_indices_buffer: Vec::with_capacity(l_capacity),
            u_values_buffer: Vec::with_capacity(u_capacity),
            u_row_indices_buffer: Vec::with_capacity(u_capacity),
            n,
        }
    }

    /// Create a workspace for simple (non-symbolic) factorization.
    ///
    /// Uses conservative estimates for buffer sizes since symbolic info isn't available.
    pub fn new_simple(n: usize, estimated_nnz: usize) -> Self {
        // For simple factorization, estimate fill-in at 2-3x original nnz
        let capacity = estimated_nnz * 3;

        Self {
            work: vec![0.0; n],
            row_perm: (0..n).collect(),
            row_perm_inv: (0..n).collect(),
            visited: vec![false; n],
            reach_stack: Vec::with_capacity(n),
            l_values_buffer: Vec::with_capacity(capacity),
            l_row_indices_buffer: Vec::with_capacity(capacity),
            u_values_buffer: Vec::with_capacity(capacity),
            u_row_indices_buffer: Vec::with_capacity(capacity),
            n,
        }
    }

    /// Clear all buffers to prepare for a new factorization.
    ///
    /// This resets all vectors to their initial state without deallocating memory.
    pub fn clear(&mut self) {
        // Reset work vector to zeros
        self.work.fill(0.0);

        // Reset permutations to identity
        for i in 0..self.n {
            self.row_perm[i] = i;
            self.row_perm_inv[i] = i;
        }

        // Clear visited flags
        self.visited.fill(false);

        // Clear reach stack
        self.reach_stack.clear();

        // Clear L and U buffers (keep capacity)
        self.l_values_buffer.clear();
        self.l_row_indices_buffer.clear();
        self.u_values_buffer.clear();
        self.u_row_indices_buffer.clear();
    }

    /// Returns the matrix dimension this workspace was created for.
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Check if this workspace is compatible with the given dimension.
    pub fn is_compatible(&self, n: usize) -> bool {
        self.n == n
    }
}
