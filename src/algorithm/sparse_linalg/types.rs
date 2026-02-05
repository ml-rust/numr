//! Types for sparse linear algebra algorithms
//!
//! Contains decomposition results, options, and diagnostic types.

use crate::runtime::Runtime;
use crate::sparse::CsrData;
use std::collections::HashMap;

// ============================================================================
// ILU(0) Types
// ============================================================================

/// Result of incomplete LU factorization
///
/// Contains the L and U factors in CSR format, where:
/// - L is unit lower triangular (ones on diagonal, stored implicitly)
/// - U is upper triangular (diagonal stored explicitly)
#[derive(Debug, Clone)]
pub struct IluDecomposition<R: Runtime> {
    /// Lower triangular factor (unit diagonal, stored implicitly as zeros)
    /// Shape: [n, n] in CSR format
    pub l: CsrData<R>,

    /// Upper triangular factor (includes diagonal)
    /// Shape: [n, n] in CSR format
    pub u: CsrData<R>,
}

/// Configuration for ILU factorization
#[derive(Debug, Clone)]
pub struct IluOptions {
    /// Tolerance for dropping small values (default: 0.0, keep all)
    pub drop_tolerance: f64,

    /// Modification for diagonal stability (default: 0.0)
    /// Adds this value to diagonal if pivot becomes too small
    pub diagonal_shift: f64,
}

impl Default for IluOptions {
    fn default() -> Self {
        Self {
            drop_tolerance: 0.0,
            diagonal_shift: 0.0,
        }
    }
}

/// Symbolic factorization for ILU(0)
///
/// Precomputed data structure that allows fast numeric ILU(0) factorization
/// when the sparsity pattern is known to be unchanged.
///
/// # Usage Pattern
///
/// ```ignore
/// // First solve: compute symbolic factorization
/// let symbolic = client.ilu0_symbolic(&pattern)?;
///
/// // Repeated solves: only compute numeric phase
/// for iteration in 0..max_iters {
///     let jacobian = compute_jacobian(&x);  // Same pattern, different values
///     let ilu = client.ilu0_numeric(&jacobian, &symbolic, opts)?;
///     let precond = |v| apply_ilu_preconditioner(&ilu, v);
///     x = gmres_solve(&jacobian, &rhs, precond)?;
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SymbolicIlu0 {
    /// Matrix dimension
    pub n: usize,

    /// Row pointers for L (same as lower triangle of input)
    pub l_row_ptrs: Vec<i64>,

    /// Column indices for L
    pub l_col_indices: Vec<i64>,

    /// Row pointers for U (same as upper triangle of input)
    pub u_row_ptrs: Vec<i64>,

    /// Column indices for U
    pub u_col_indices: Vec<i64>,

    /// Position of diagonal element in each row of the combined LU storage
    /// Used for fast diagonal access during numeric factorization
    pub diag_positions: Vec<usize>,

    /// For each row, list of (k, idx_ik, update_list) where:
    /// - k is a column index < i
    /// - idx_ik is the position of L[i,k] in values
    /// - update_list contains (j, idx_ij, idx_kj) for the update A[i,j] -= L[i,k]*U[k,j]
    ///
    /// This precomputed structure avoids hash map lookups during numeric factorization.
    pub update_schedule: Vec<Vec<(usize, usize, Vec<(usize, usize, usize)>)>>,
}

// ============================================================================
// IC(0) Types
// ============================================================================

/// Result of incomplete Cholesky factorization
///
/// Contains the L factor such that A ≈ L·Lᵀ.
#[derive(Debug, Clone)]
pub struct IcDecomposition<R: Runtime> {
    /// Lower triangular factor
    /// Shape: [n, n] in CSR format
    pub l: CsrData<R>,
}

/// Configuration for IC factorization
#[derive(Debug, Clone)]
pub struct IcOptions {
    /// Tolerance for dropping small values (default: 0.0, keep all)
    pub drop_tolerance: f64,

    /// Shift to add to diagonal for positive definiteness (default: 0.0)
    pub diagonal_shift: f64,
}

impl Default for IcOptions {
    fn default() -> Self {
        Self {
            drop_tolerance: 0.0,
            diagonal_shift: 0.0,
        }
    }
}

// ============================================================================
// ILU(k) Types
// ============================================================================

/// Fill level for ILU(k) factorization
///
/// ILU(k) allows fill-in at positions where the level-of-fill ≤ k.
/// Higher levels give better approximations but use more memory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum IluFillLevel {
    /// ILU(0): Zero fill-in, same sparsity as A
    #[default]
    Zero,
    /// ILU(1): Level-1 fill-in
    One,
    /// ILU(2): Level-2 fill-in
    Two,
    /// ILU(3): Level-3 fill-in
    Three,
}

impl IluFillLevel {
    /// Get the numeric level value
    pub fn level(&self) -> usize {
        match self {
            IluFillLevel::Zero => 0,
            IluFillLevel::One => 1,
            IluFillLevel::Two => 2,
            IluFillLevel::Three => 3,
        }
    }

    /// Create from numeric level (capped at 3)
    pub fn from_level(level: usize) -> Self {
        match level {
            0 => IluFillLevel::Zero,
            1 => IluFillLevel::One,
            2 => IluFillLevel::Two,
            _ => IluFillLevel::Three,
        }
    }

    /// Upgrade to next fill level, if possible
    pub fn upgrade(&self) -> Option<IluFillLevel> {
        match self {
            IluFillLevel::Zero => Some(IluFillLevel::One),
            IluFillLevel::One => Some(IluFillLevel::Two),
            IluFillLevel::Two => Some(IluFillLevel::Three),
            IluFillLevel::Three => None,
        }
    }
}

impl std::fmt::Display for IluFillLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ILU({})", self.level())
    }
}

/// Configuration for ILU(k) factorization
#[derive(Debug, Clone)]
pub struct IlukOptions {
    /// Fill level (0, 1, 2, or 3)
    pub fill_level: IluFillLevel,

    /// Tolerance for dropping small values (default: 0.0, keep all)
    pub drop_tolerance: f64,

    /// Modification for diagonal stability (default: 0.0)
    pub diagonal_shift: f64,

    /// Threshold for pivot selection (default: 0.1)
    /// Pivots below this threshold relative to the row max are shifted
    pub pivot_threshold: f64,
}

impl Default for IlukOptions {
    fn default() -> Self {
        Self {
            fill_level: IluFillLevel::Zero,
            drop_tolerance: 0.0,
            diagonal_shift: 0.0,
            pivot_threshold: 0.1,
        }
    }
}

/// Metrics from ILU factorization for diagnostics
#[derive(Debug, Clone)]
pub struct IluMetrics {
    /// Number of non-zeros in original matrix
    pub original_nnz: usize,

    /// Number of non-zeros in factored L+U
    pub factored_nnz: usize,

    /// Ratio: factored_nnz / original_nnz
    pub fill_ratio: f64,

    /// Fill level used
    pub fill_level: IluFillLevel,

    /// Number of diagonal shifts applied
    pub diagonal_shifts_applied: usize,
}

/// Result of ILU(k) factorization
#[derive(Debug, Clone)]
pub struct IlukDecomposition<R: Runtime> {
    /// Lower triangular factor (unit diagonal, stored implicitly as zeros)
    pub l: CsrData<R>,

    /// Upper triangular factor (includes diagonal)
    pub u: CsrData<R>,

    /// Factorization metrics
    pub metrics: IluMetrics,
}

/// Symbolic factorization result for ILU(k)
///
/// Stores the fill pattern computed during symbolic analysis.
/// Can be reused for multiple numeric factorizations with different values
/// but the same sparsity pattern.
#[derive(Debug, Clone)]
pub struct IlukSymbolic {
    /// Matrix dimension
    pub n: usize,

    /// Level of fill for each (row, col) position
    /// level_of_fill[row] maps col -> level
    pub level_of_fill: Vec<HashMap<usize, usize>>,

    /// Row pointers for L factor
    pub row_ptrs_l: Vec<i64>,

    /// Column indices for L factor
    pub col_indices_l: Vec<i64>,

    /// Row pointers for U factor
    pub row_ptrs_u: Vec<i64>,

    /// Column indices for U factor
    pub col_indices_u: Vec<i64>,

    /// Fill level used
    pub fill_level: IluFillLevel,
}
