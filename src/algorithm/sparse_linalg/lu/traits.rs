//! Trait definitions for sparse LU factorization kernels
//!
//! Defines the contract that all backends must implement for sparse LU
//! numeric factorization using the Gilbert-Peierls algorithm.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::sparse::CscData;

use super::types::{LuFactors, LuMetrics, LuOptions, LuSymbolic};

/// Sparse LU factorization algorithms
///
/// This trait defines the numeric factorization algorithms that backends must
/// implement. The symbolic analysis (elimination tree, reach computation)
/// is performed by solvr and passed in as `LuSymbolic`.
///
/// # Algorithm: Gilbert-Peierls Left-Looking LU
///
/// ```text
/// For each column k = 0 to n-1:
///   1. Initialize work vector: x = A[:, k]
///   2. For each j in reach(k) in topological order:
///      a. x[j] = x[j] / L[j,j]           (divide by pivot)
///      b. For i in L[j+1:, j]:
///         x[i] = x[i] - L[i,j] * x[j]    (sparse AXPY)
///   3. Find pivot: p = argmax_{i >= k} |x[i]|
///   4. Swap rows k and p in x and permutation
///   5. Store: L[k+1:, k] = x[k+1:] / x[k]
///            U[0:k+1, k] = x[0:k+1]
/// ```
///
/// # Kernel Operations
///
/// The algorithm consists of these kernel-worthy operations:
/// - **Scatter**: Copy sparse column into dense work vector
/// - **Sparse triangular solve**: Steps 2a-2b above (main computational work)
/// - **Pivot search**: Find maximum absolute value (SIMD reduction)
/// - **Gather**: Extract nonzeros from work vector to sparse storage
///
/// # Implementation Requirements
///
/// All backends must produce numerically identical results (within tolerance):
/// - Same pivot selection (using same tolerance)
/// - Same order of operations in sparse triangular solve
/// - Same handling of small/zero pivots
pub trait SparseLuOps<R: Runtime> {
    /// Sparse LU factorization with full symbolic information
    ///
    /// # Arguments
    ///
    /// * `a` - Square sparse matrix in CSC format
    /// * `symbolic` - Precomputed symbolic factorization from solvr
    /// * `options` - Factorization options (pivoting, tolerances)
    ///
    /// # Returns
    ///
    /// LU factors with row permutation from partial pivoting
    ///
    /// # Errors
    ///
    /// - Matrix is not square
    /// - Symbolic structure doesn't match matrix pattern
    /// - Zero pivot encountered (and no diagonal shift specified)
    fn sparse_lu(
        &self,
        a: &CscData<R>,
        symbolic: &LuSymbolic,
        options: &LuOptions,
    ) -> Result<LuFactors<R>>;

    /// Sparse LU factorization with simple symbolic structure
    ///
    /// Uses the matrix pattern directly without full symbolic analysis.
    /// Fill-in is discovered dynamically, which is less efficient but
    /// works without solvr's symbolic analysis.
    ///
    /// # Arguments
    ///
    /// * `a` - Square sparse matrix in CSC format
    /// * `options` - Factorization options
    ///
    /// # Returns
    ///
    /// LU factors with row permutation
    fn sparse_lu_simple(&self, a: &CscData<R>, options: &LuOptions) -> Result<LuFactors<R>>;

    /// Solve Ax = b using precomputed LU factors
    ///
    /// Solves by: x = U⁻¹ L⁻¹ P b
    ///
    /// # Arguments
    ///
    /// * `factors` - Precomputed LU factorization
    /// * `b` - Right-hand side vector `[n]` or matrix `[n, nrhs]`
    ///
    /// # Returns
    ///
    /// Solution vector x
    fn sparse_lu_solve(
        &self,
        factors: &LuFactors<R>,
        b: &crate::tensor::Tensor<R>,
    ) -> Result<crate::tensor::Tensor<R>>;

    /// Get factorization metrics (for diagnostics)
    fn sparse_lu_with_metrics(
        &self,
        a: &CscData<R>,
        symbolic: &LuSymbolic,
        options: &LuOptions,
    ) -> Result<(LuFactors<R>, LuMetrics)>;
}

/// Low-level kernel operations for sparse LU
///
/// These are the primitive operations that benefit from SIMD/GPU acceleration.
/// Higher-level factorization algorithms compose these kernels.
pub trait SparseLuKernels<R: Runtime> {
    /// Scatter sparse column into dense work vector
    ///
    /// `work[row_indices[i]]` = `values[i]` for i in 0..nnz
    ///
    /// # Arguments
    ///
    /// * `values` - Sparse column values
    /// * `row_indices` - Row indices for each value
    /// * `work` - Dense work vector (length n), modified in place
    fn scatter_column(&self, values: &[f64], row_indices: &[i64], work: &mut [f64]);

    /// Sparse AXPY: `work[indices]` -= scale * values
    ///
    /// For each i: `work[row_indices[i]]` -= scale * `values[i]`
    ///
    /// # Arguments
    ///
    /// * `scale` - Scalar multiplier
    /// * `values` - Sparse vector values
    /// * `row_indices` - Row indices for each value
    /// * `work` - Dense work vector, modified in place
    fn sparse_axpy(&self, scale: f64, values: &[f64], row_indices: &[i64], work: &mut [f64]);

    /// Find index of maximum absolute value in `work[start..end]`
    ///
    /// Returns (index, max_abs_value) where index is in `[start, end)`
    ///
    /// # Arguments
    ///
    /// * `work` - Dense work vector
    /// * `start` - Start index (inclusive)
    /// * `end` - End index (exclusive)
    fn find_pivot(&self, work: &[f64], start: usize, end: usize) -> (usize, f64);

    /// Gather nonzeros from work vector into sparse output
    ///
    /// `output[i]` = `work[row_indices[i]]` for i in 0..nnz
    /// Then: `work[row_indices[i]]` = 0.0 (clear for next column)
    ///
    /// # Arguments
    ///
    /// * `work` - Dense work vector, cleared after gather
    /// * `row_indices` - Row indices to gather
    /// * `output` - Output sparse values
    fn gather_and_clear(&self, work: &mut [f64], row_indices: &[i64], output: &mut [f64]);

    /// Divide work vector elements by pivot
    ///
    /// `work[row_indices[i]] /= pivot` for `i` in `0..nnz`
    ///
    /// # Arguments
    ///
    /// * `work` - Dense work vector, modified in place
    /// * `row_indices` - Row indices to divide
    /// * `pivot` - Pivot value to divide by
    fn divide_by_pivot(&self, work: &mut [f64], row_indices: &[i64], pivot: f64);

    /// Swap two rows in permutation and work vector
    ///
    /// # Arguments
    ///
    /// * `work` - Dense work vector
    /// * `perm` - Row permutation array
    /// * `row_a` - First row index
    /// * `row_b` - Second row index
    fn swap_rows(&self, work: &mut [f64], perm: &mut [usize], row_a: usize, row_b: usize);
}

// ============================================================================
// Validation helpers
// ============================================================================

/// Validate that matrix pattern matches symbolic structure
pub fn validate_symbolic_pattern(
    col_ptrs: &[i64],
    _row_indices: &[i64],
    symbolic: &LuSymbolic,
) -> Result<()> {
    let n = symbolic.n;

    if col_ptrs.len() != n + 1 {
        return Err(crate::error::Error::ShapeMismatch {
            expected: vec![n + 1],
            got: vec![col_ptrs.len()],
        });
    }

    // Check that matrix nnz is compatible with symbolic
    let matrix_nnz = col_ptrs[n] as usize;
    let symbolic_nnz = symbolic.l_nnz() + symbolic.u_nnz();

    if matrix_nnz > symbolic_nnz {
        return Err(crate::error::Error::Internal(format!(
            "Matrix has more nonzeros ({}) than symbolic structure allows ({})",
            matrix_nnz, symbolic_nnz
        )));
    }

    Ok(())
}

/// Validate dimensions for LU solve
pub fn validate_lu_solve_dims(
    factors: &LuFactors<impl Runtime>,
    b_shape: &[usize],
) -> Result<(usize, usize)> {
    let n = factors.row_perm.len();

    if b_shape.is_empty() {
        return Err(crate::error::Error::Internal(
            "Right-hand side must be at least 1D".to_string(),
        ));
    }

    if b_shape[0] != n {
        return Err(crate::error::Error::ShapeMismatch {
            expected: vec![n],
            got: vec![b_shape[0]],
        });
    }

    let nrhs = if b_shape.len() == 1 { 1 } else { b_shape[1] };

    Ok((n, nrhs))
}
