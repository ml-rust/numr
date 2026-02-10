//! Sparse linear algebra algorithm trait
//!
//! Defines the contract that all backends must implement for sparse
//! linear algebra operations used in iterative solvers.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

use super::types::{
    IcDecomposition, IcOptions, IluDecomposition, IluFillLevel, IluOptions, IlukDecomposition,
    IlukOptions, IlukSymbolic, SymbolicIlu0,
};

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
    ///   For k = 0 to i-1 where `` `A[i,k]` `` ≠ 0:
    ///     `` `A[i,k] = A[i,k] / A[k,k]` ``  (L factor)
    ///     For j = k+1 to n-1 where `` `A[i,j]` `` ≠ 0 AND `` `A[k,j]` `` ≠ 0:
    ///       `` `A[i,j] = A[i,j] - A[i,k] * A[k,j]` ``  (U factor update)
    ///
    /// Result:
    ///   `` `L[i,k] = A[i,k]` `` for k < i
    ///   `` `U[i,j] = A[i,j]` `` for j >= i
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
    ///   For k = 0 to i-1 where `` `A[i,k]` `` ≠ 0:
    ///     For j = 0 to k-1 where `` `A[i,j]` `` ≠ 0 AND `` `L[k,j]` `` ≠ 0:
    ///       `` `A[i,k] = A[i,k] - A[i,j] * L[k,j]` ``
    ///     `` `A[i,k] = A[i,k] / L[k,k]` ``
    ///
    ///   `` `sum = A[i,i]` ``
    ///   For j = 0 to i-1 where `` `A[i,j]` `` ≠ 0:
    ///     `` `sum = sum - A[i,j]²` ``
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
    ///   `` `x[i] = b[i]` ``
    ///   For j in nonzeros(`` `L[i, 0:i]` ``):
    ///     `` `x[i] = x[i] - L[i,j] * x[j]` ``
    ///   `` `x[i] = x[i] / L[i,i]` ``  (if not unit diagonal)
    /// ```
    ///
    /// **Upper triangular (backward substitution):**
    /// ```text
    /// For i = n-1 down to 0:
    ///   `x[i]` = `b[i]`
    ///   For j in nonzeros(`U[i, i+1:n]`):
    ///     `x[i]` = `x[i]` - `U[i,j]` * `x[j]`
    ///   `x[i]` = `x[i]` / `U[i,i]`
    /// ```
    ///
    /// # Arguments
    ///
    /// * `l_or_u` - Sparse triangular matrix in CSR format
    /// * `b` - Right-hand side dense vector `` `[n]` `` or matrix `` `[n, k]` ``
    /// * `lower` - True for lower triangular, false for upper triangular
    /// * `unit_diagonal` - True if diagonal is implicitly 1 (for unit triangular)
    ///
    /// # Returns
    ///
    /// Solution vector x `` `[n]` `` or matrix `` `[n, k]` ``
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

    // ========================================================================
    // ILU(k) Methods
    // ========================================================================

    /// ILU(k) symbolic factorization: compute fill pattern
    ///
    /// Analyzes the sparsity structure of A to determine which positions will
    /// have non-zero values in L and U for the given fill level.
    ///
    /// # Algorithm (Level-of-Fill)
    ///
    /// ```text
    /// Initialize: level[i,j] = 0 if A[i,j] ≠ 0, else infinity
    ///
    /// For i = 0 to n-1:
    ///   For k = 0 to i-1 where level[i,k] ≤ fill_level:
    ///     For j = k+1 to n-1 where level[k,j] ≤ fill_level:
    ///       new_level = level[i,k] + level[k,j] + 1
    ///       if new_level ≤ fill_level:
    ///         level[i,j] = min(level[i,j], new_level)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Square sparse matrix (only pattern is used, not values)
    /// * `level` - Fill level (0, 1, 2, or 3)
    ///
    /// # Returns
    ///
    /// Symbolic factorization containing the fill pattern
    fn iluk_symbolic(&self, a: &CsrData<R>, level: IluFillLevel) -> Result<IlukSymbolic>;

    /// ILU(k) numeric factorization: compute values using precomputed pattern
    ///
    /// # Arguments
    ///
    /// * `a` - Square sparse matrix (must have same pattern as used for symbolic)
    /// * `symbolic` - Precomputed symbolic factorization
    /// * `opts` - Factorization options
    ///
    /// # Returns
    ///
    /// ILU(k) decomposition with L and U factors
    fn iluk_numeric(
        &self,
        a: &CsrData<R>,
        symbolic: &IlukSymbolic,
        opts: &IlukOptions,
    ) -> Result<IlukDecomposition<R>>;

    /// ILU(k) factorization (combined symbolic + numeric)
    ///
    /// Convenience method that performs both phases. For repeated solves
    /// with the same pattern, use `iluk_symbolic` + `iluk_numeric` instead.
    ///
    /// # Arguments
    ///
    /// * `a` - Square sparse matrix
    /// * `opts` - Factorization options including fill level
    ///
    /// # Returns
    ///
    /// ILU(k) decomposition with L and U factors
    fn iluk(&self, a: &CsrData<R>, opts: IlukOptions) -> Result<IlukDecomposition<R>>;

    // ========================================================================
    // ILU(0) Symbolic/Numeric Split (for pattern reuse)
    // ========================================================================

    /// ILU(0) symbolic factorization: precompute update schedule
    ///
    /// Analyzes the sparsity pattern to create an efficient update schedule
    /// for numeric factorization. This avoids hash map lookups during the
    /// numeric phase.
    ///
    /// # Arguments
    ///
    /// * `pattern` - Sparse matrix whose pattern will be used
    ///
    /// # Returns
    ///
    /// Symbolic factorization with precomputed update schedule
    fn ilu0_symbolic(&self, pattern: &CsrData<R>) -> Result<SymbolicIlu0>;

    /// ILU(0) numeric factorization using precomputed symbolic data
    ///
    /// # Arguments
    ///
    /// * `a` - Sparse matrix (must have same pattern as used for symbolic)
    /// * `symbolic` - Precomputed symbolic factorization
    /// * `options` - Factorization options
    ///
    /// # Returns
    ///
    /// ILU decomposition with L and U factors
    fn ilu0_numeric(
        &self,
        a: &CsrData<R>,
        symbolic: &SymbolicIlu0,
        options: IluOptions,
    ) -> Result<IluDecomposition<R>>;
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
