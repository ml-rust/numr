//! Sparse Linear Algebra Algorithm Contracts
//!
//! This module defines traits for sparse linear algebra algorithms that are
//! essential for iterative solvers. All backends must implement the same
//! algorithms to ensure numerical parity.
//!
//! # Algorithms
//!
//! - **ILU(0)**: Incomplete LU factorization with zero fill-in
//! - **IC(0)**: Incomplete Cholesky factorization with zero fill-in
//! - **Sparse Triangular Solve**: Forward/backward substitution for sparse triangular matrices
//!
//! # Use Cases
//!
//! These algorithms are primarily used as **preconditioners** for iterative solvers:
//!
//! - **ILU(0)** → Preconditioner for GMRES, BiCGSTAB (non-symmetric systems)
//! - **IC(0)** → Preconditioner for CG (symmetric positive definite systems)
//! - **Sparse Triangular Solve** → Application of the preconditioner (L⁻¹ or U⁻¹)
//!
//! # Mathematical Background
//!
//! ## Incomplete LU (ILU)
//!
//! For a sparse matrix A, ILU computes approximate factors L and U such that:
//! - A ≈ L·U
//! - sparsity(L) ⊆ sparsity(lower(A))
//! - sparsity(U) ⊆ sparsity(upper(A))
//!
//! ILU(0) means zero fill-in: L and U have the same sparsity pattern as A.
//!
//! ## Incomplete Cholesky (IC)
//!
//! For a symmetric positive definite sparse matrix A, IC computes L such that:
//! - A ≈ L·Lᵀ
//! - sparsity(L) ⊆ sparsity(lower(A))
//!
//! IC(0) means zero fill-in: L has the same sparsity pattern as lower(A).
//!
//! # Algorithm Guarantees
//!
//! All backends produce identical results (within floating-point tolerance):
//! - Same sparsity pattern in output
//! - Same numerical values (order of operations matches)
//! - Same handling of near-zero pivots

use crate::error::Result;
use crate::runtime::Runtime;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

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

/// Result of incomplete Cholesky factorization
///
/// Contains the L factor such that A ≈ L·Lᵀ.
#[derive(Debug, Clone)]
pub struct IcDecomposition<R: Runtime> {
    /// Lower triangular factor
    /// Shape: [n, n] in CSR format
    pub l: CsrData<R>,
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

/// Sparse linear algebra algorithms for preconditioning
///
/// This trait defines the algorithms that all backends must implement
/// for sparse matrix factorizations used in iterative solvers.
///
/// # Implementation Requirements
///
/// Backends may differ in:
/// - Parallelization strategy (row-parallel, level-scheduled, etc.)
/// - Memory access patterns
///
/// Backends MUST match in:
/// - Mathematical formula (same row-by-row factorization)
/// - Order of operations (left-looking vs right-looking must be consistent)
/// - Handling of zero/small pivots
/// - Output sparsity pattern (zero fill-in guarantee)
pub trait SparseLinAlgAlgorithms<R: Runtime> {
    /// Incomplete LU factorization with zero fill-in: A ≈ L·U
    ///
    /// # Algorithm (IKJ variant, row-by-row)
    ///
    /// ```text
    /// For i = 0 to n-1:
    ///   For k = 0 to i-1 where A[i,k] ≠ 0:
    ///     A[i,k] = A[i,k] / A[k,k]  (L factor)
    ///     For j = k+1 to n-1 where A[i,j] ≠ 0 AND A[k,j] ≠ 0:
    ///       A[i,j] = A[i,j] - A[i,k] * A[k,j]  (U factor update)
    ///
    /// Result:
    ///   L[i,k] = A[i,k] for k < i
    ///   U[i,j] = A[i,j] for j >= i
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Square sparse matrix in CSR format
    /// * `options` - Factorization options (drop tolerance, diagonal shift)
    ///
    /// # Returns
    ///
    /// ILU decomposition with L (unit lower triangular) and U (upper triangular)
    ///
    /// # Errors
    ///
    /// - Matrix is not square
    /// - Zero pivot encountered (and no diagonal shift specified)
    fn ilu0(&self, a: &CsrData<R>, options: IluOptions) -> Result<IluDecomposition<R>>;

    /// Incomplete Cholesky factorization with zero fill-in: A ≈ L·Lᵀ
    ///
    /// # Algorithm (row-by-row Cholesky)
    ///
    /// ```text
    /// For i = 0 to n-1:
    ///   For k = 0 to i-1 where A[i,k] ≠ 0:
    ///     For j = 0 to k-1 where A[i,j] ≠ 0 AND L[k,j] ≠ 0:
    ///       A[i,k] = A[i,k] - A[i,j] * L[k,j]
    ///     A[i,k] = A[i,k] / L[k,k]
    ///
    ///   sum = A[i,i]
    ///   For j = 0 to i-1 where A[i,j] ≠ 0:
    ///     sum = sum - A[i,j]²
    ///   L[i,i] = sqrt(sum)
    ///
    /// Result: L lower triangular where A ≈ L·Lᵀ
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric positive definite sparse matrix in CSR format
    /// * `options` - Factorization options
    ///
    /// # Returns
    ///
    /// IC decomposition with lower triangular factor L
    ///
    /// # Errors
    ///
    /// - Matrix is not square
    /// - Matrix is not positive definite (negative diagonal encountered)
    ///
    /// # Note
    ///
    /// The input matrix should be symmetric. Only the lower triangle is used.
    fn ic0(&self, a: &CsrData<R>, options: IcOptions) -> Result<IcDecomposition<R>>;

    /// Sparse triangular solve: solve L·x = b or U·x = b
    ///
    /// # Algorithm
    ///
    /// **Lower triangular (forward substitution):**
    /// ```text
    /// For i = 0 to n-1:
    ///   x[i] = b[i]
    ///   For j in nonzeros(L[i, 0:i]):
    ///     x[i] = x[i] - L[i,j] * x[j]
    ///   x[i] = x[i] / L[i,i]  (if not unit diagonal)
    /// ```
    ///
    /// **Upper triangular (backward substitution):**
    /// ```text
    /// For i = n-1 down to 0:
    ///   x[i] = b[i]
    ///   For j in nonzeros(U[i, i+1:n]):
    ///     x[i] = x[i] - U[i,j] * x[j]
    ///   x[i] = x[i] / U[i,i]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `l_or_u` - Sparse triangular matrix in CSR format
    /// * `b` - Right-hand side dense vector [n] or matrix [n, k]
    /// * `lower` - True for lower triangular, false for upper triangular
    /// * `unit_diagonal` - True if diagonal is implicitly 1 (for unit triangular)
    ///
    /// # Returns
    ///
    /// Solution vector x [n] or matrix [n, k]
    ///
    /// # Errors
    ///
    /// - Dimension mismatch between matrix and vector
    /// - Zero diagonal encountered (when not unit diagonal)
    fn sparse_solve_triangular(
        &self,
        l_or_u: &CsrData<R>,
        b: &Tensor<R>,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Tensor<R>>;
}

// ============================================================================
// Validation helpers
// ============================================================================

/// Validate that matrix is square for factorization
pub fn validate_square_sparse(shape: [usize; 2]) -> Result<usize> {
    let [nrows, ncols] = shape;
    if nrows != ncols {
        return Err(crate::error::Error::ShapeMismatch {
            expected: vec![nrows, nrows],
            got: vec![nrows, ncols],
        });
    }
    Ok(nrows)
}

/// Validate dimensions for triangular solve
pub fn validate_triangular_solve_dims(
    matrix_shape: [usize; 2],
    b_shape: &[usize],
) -> Result<(usize, usize)> {
    use crate::error::Error;

    let n = validate_square_sparse(matrix_shape)?;

    if b_shape.is_empty() {
        return Err(Error::Internal(
            "Right-hand side must be at least 1D".to_string(),
        ));
    }

    if b_shape[0] != n {
        return Err(Error::ShapeMismatch {
            expected: vec![n],
            got: vec![b_shape[0]],
        });
    }

    let nrhs = if b_shape.len() == 1 { 1 } else { b_shape[1] };

    Ok((n, nrhs))
}
