//! Types for sparse QR factorization
//!
//! Contains factorization results, symbolic structures, and options.

use crate::runtime::Runtime;
use crate::sparse::CscData;
use crate::tensor::Tensor;

// ============================================================================
// QR Factorization Types
// ============================================================================

/// Result of sparse Householder QR factorization: A*P = Q*R
///
/// Q is stored implicitly as a sequence of Householder reflectors.
/// R is stored explicitly in CSC format.
/// P is the column permutation from COLAMD ordering.
///
/// For GPU backends, Householder vectors and tau are stored GPU-resident only
/// (`gpu_householder_values`, `gpu_tau`), and the CPU sparse fields
/// (`householder_vectors`, `tau`) are empty. GPU solve uses the GPU tensors
/// directly. CPU factorization populates the CPU fields instead.
#[derive(Debug, Clone)]
pub struct QrFactors<R: Runtime> {
    /// Householder reflectors stored as sparse vectors (CPU).
    /// Each entry is (row_indices, values) for one reflector.
    /// Reflector k has support in rows k..m.
    /// Empty for GPU-factorized results (use `gpu_householder_values` instead).
    pub householder_vectors: Vec<(Vec<i64>, Vec<f64>)>,

    /// Tau coefficients for each Householder reflector.
    /// `Q_k = I - tau_k * v_k * v_k^T`
    /// Empty for GPU-factorized results (use `gpu_tau` instead).
    pub tau: Vec<f64>,

    /// Upper triangular factor R in CSC format.
    /// Shape: `[m, n]` but only first `rank` rows of each column are meaningful.
    pub r: CscData<R>,

    /// Column permutation from COLAMD ordering.
    /// `col_perm[k]` = original column index of the k-th column in the permuted matrix.
    pub col_perm: Vec<usize>,

    /// Numerical rank detected during factorization.
    pub rank: usize,

    /// Dense Householder vectors on GPU (optional, for GPU-resident solve).
    ///
    /// Flat buffer of length `sum(m-k for k in 0..min(m,n))`. Reflector k is
    /// stored at `h_offset(k, m)` with length `m - k`. Only populated by GPU
    /// factorization backends; `None` for CPU factorization.
    pub gpu_householder_values: Option<Tensor<R>>,

    /// Tau coefficients on GPU (optional, for GPU-resident solve).
    ///
    /// Length `min(m, n)`. Only populated by GPU factorization backends.
    pub gpu_tau: Option<Tensor<R>>,
}

/// Symbolic analysis for sparse QR factorization
///
/// Precomputed structural information based on the sparsity pattern.
/// Reusable for multiple numeric factorizations with the same pattern.
#[derive(Debug, Clone)]
pub struct QrSymbolic {
    /// Number of rows
    pub m: usize,

    /// Number of columns
    pub n: usize,

    /// Elimination tree for R: `etree[j]` = parent of column j, or -1 if root.
    /// Derived from the column structure of A^T*A without forming it explicitly.
    pub etree: Vec<i64>,

    /// Predicted column counts for R (upper bound on nnz per column).
    pub r_col_counts: Vec<usize>,

    /// Column permutation from COLAMD.
    pub col_perm: Vec<usize>,

    /// Predicted total nnz in R.
    pub predicted_r_nnz: usize,
}

impl QrSymbolic {
    /// Create a trivial symbolic structure (identity permutation, no etree).
    pub fn identity(m: usize, n: usize) -> Self {
        Self {
            m,
            n,
            etree: vec![-1; n],
            r_col_counts: vec![1; n],
            col_perm: (0..n).collect(),
            predicted_r_nnz: n,
        }
    }
}

/// Configuration for sparse QR factorization
#[derive(Debug, Clone)]
pub struct QrOptions {
    /// Tolerance for rank detection (default: 1e-12).
    /// Diagonal entries of R with absolute value below this are treated as zero.
    pub rank_tolerance: f64,

    /// Column ordering strategy.
    pub ordering: QrOrdering,
}

impl Default for QrOptions {
    fn default() -> Self {
        Self {
            rank_tolerance: 1e-12,
            ordering: QrOrdering::Colamd,
        }
    }
}

impl QrOptions {
    /// Create options with no column ordering.
    pub fn no_ordering() -> Self {
        Self {
            ordering: QrOrdering::Identity,
            ..Default::default()
        }
    }
}

/// Column ordering strategy for QR factorization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QrOrdering {
    /// No column permutation (identity permutation, original column order).
    Identity,
    /// COLAMD approximate minimum degree ordering.
    Colamd,
}

/// Metrics from QR factorization for diagnostics
#[derive(Debug, Clone)]
pub struct QrMetrics {
    /// Number of non-zeros in original matrix
    pub original_nnz: usize,

    /// Number of non-zeros in R factor
    pub r_nnz: usize,

    /// Fill ratio: r_nnz / original_nnz
    pub fill_ratio: f64,

    /// Numerical rank detected
    pub numerical_rank: usize,
}
