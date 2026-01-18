//! Linear algebra algorithm contracts for backend consistency
//!
//! This module defines traits that ensure all backends implement the same
//! mathematical algorithms for linear algebra operations. This guarantees
//! numerical parity across CPU, CUDA, WebGPU, and other backends.
//!
//! # Design Principles
//!
//! 1. **Algorithm-Level Contract**: Each trait method represents a specific algorithm
//! 2. **Backend Parity**: Same algorithm must produce same results (within FP tolerance)
//! 3. **Explicit Contracts**: Missing implementations cause compile errors
//! 4. **Testability**: Easy to verify all backends implement the same algorithm
//!
//! # Why Not Vendor Libraries?
//!
//! numr must work WITHOUT cuSOLVER/MKL/LAPACK. Native implementations are required.
//! Vendor libraries may be used as optional optimizations behind `vendor-blas` feature.

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

// ============================================================================
// LU Decomposition
// ============================================================================

/// LU decomposition result: PA = LU
///
/// L is lower triangular with unit diagonal, U is upper triangular.
/// P is a permutation matrix (stored as pivot indices).
pub struct LuDecomposition<R: Runtime> {
    /// Lower triangular factor L (unit diagonal, stored in lower part)
    /// Upper triangular factor U (stored in upper part including diagonal)
    /// Stored together in a single matrix for memory efficiency.
    pub lu: Tensor<R>,

    /// Pivot indices: row i was swapped with row pivots[i]
    /// I64 tensor of shape [min(m, n)]
    pub pivots: Tensor<R>,

    /// Number of row swaps (for determinant sign)
    pub num_swaps: usize,
}

/// Cholesky decomposition result: A = LL^T
///
/// Only valid for symmetric positive-definite matrices.
pub struct CholeskyDecomposition<R: Runtime> {
    /// Lower triangular factor L such that A = L @ L^T
    pub l: Tensor<R>,
}

/// QR decomposition result: A = QR
///
/// Q is orthogonal (Q^T @ Q = I), R is upper triangular.
pub struct QrDecomposition<R: Runtime> {
    /// Orthogonal matrix Q [m, m] or [m, k] for thin QR
    pub q: Tensor<R>,

    /// Upper triangular matrix R [m, n] or [k, n] for thin QR
    pub r: Tensor<R>,
}

/// Singular Value Decomposition result: A = U @ S @ V^T
pub struct SvdDecomposition<R: Runtime> {
    /// Left singular vectors U [m, k] where k = min(m, n)
    pub u: Tensor<R>,

    /// Singular values S [k] (sorted in descending order)
    pub s: Tensor<R>,

    /// Right singular vectors V^T [k, n]
    pub vt: Tensor<R>,
}

// ============================================================================
// LinearAlgebra Trait
// ============================================================================

/// Algorithmic contract for linear algebra operations
///
/// All backends implementing linear algebra MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
///
/// # Algorithms Defined
///
/// - **LU Decomposition (Doolittle with Partial Pivoting)**:
///   - For each column k = 0..min(m,n):
///     1. Find pivot: argmax(|A[k:m, k]|)
///     2. Swap rows k and pivot
///     3. Compute multipliers: L[i,k] = A[i,k] / A[k,k] for i > k
///     4. Eliminate: A[i,j] -= L[i,k] * A[k,j] for i,j > k
///
/// - **Cholesky Decomposition (Cholesky-Banachiewicz)**:
///   - For each row i = 0..n:
///     1. Diagonal: L[i,i] = sqrt(A[i,i] - sum(L[i,k]² for k < i))
///     2. Off-diagonal: L[j,i] = (A[j,i] - sum(L[j,k]*L[i,k] for k < i)) / L[i,i]
///
/// - **QR Decomposition (Householder Reflections)**:
///   - For each column k = 0..min(m,n):
///     1. Compute Householder vector v from A[k:m, k]
///     2. Apply reflection: A[k:m, k:n] -= 2*v*(v^T @ A[k:m, k:n])
///
/// # Implementation Requirements
///
/// Backends may differ in:
/// - Parallelization strategy (threads, SIMD, GPU blocks)
/// - Memory access patterns (blocking, tiling)
/// - Loop unrolling and compiler optimizations
///
/// Backends MUST match in:
/// - Mathematical formula (same operations in same order)
/// - Pivot selection criteria (partial pivoting for LU)
/// - Special case handling (singular matrices, non-positive-definite)
pub trait LinearAlgebraAlgorithms<R: Runtime> {
    // ========================================================================
    // Matrix Decompositions
    // ========================================================================

    /// LU Decomposition with partial pivoting: PA = LU
    ///
    /// # Algorithm (Doolittle with Partial Pivoting)
    ///
    /// ```text
    /// Input: A [m, n] matrix
    /// Output: LU [m, n] (L lower, U upper), pivots [min(m,n)]
    ///
    /// For k = 0 to min(m, n) - 1:
    ///   // Find pivot (max absolute value in column k, rows k to m-1)
    ///   pivot_row = argmax(|A[k:m, k]|) + k
    ///
    ///   // Swap rows
    ///   swap(A[k, :], A[pivot_row, :])
    ///   pivots[k] = pivot_row
    ///
    ///   // Check for zero pivot (singular matrix)
    ///   if |A[k, k]| < epsilon:
    ///     return Error::SingularMatrix
    ///
    ///   // Compute multipliers (L column)
    ///   for i = k+1 to m-1:
    ///     A[i, k] = A[i, k] / A[k, k]
    ///
    ///   // Update trailing submatrix
    ///   for i = k+1 to m-1:
    ///     for j = k+1 to n-1:
    ///       A[i, j] -= A[i, k] * A[k, j]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix [m, n]
    ///
    /// # Returns
    ///
    /// LuDecomposition containing:
    /// - `lu`: Combined L and U factors [m, n]
    /// - `pivots`: Permutation indices [min(m,n)]
    /// - `num_swaps`: Number of row swaps (for determinant sign)
    ///
    /// # Errors
    ///
    /// Returns `Error::SingularMatrix` if matrix is singular (zero pivot).
    fn lu_decompose(&self, a: &Tensor<R>) -> Result<LuDecomposition<R>>;

    /// Cholesky Decomposition: A = LL^T
    ///
    /// # Algorithm (Cholesky-Banachiewicz)
    ///
    /// ```text
    /// Input: A [n, n] symmetric positive-definite matrix
    /// Output: L [n, n] lower triangular
    ///
    /// For i = 0 to n-1:
    ///   // Compute diagonal element
    ///   sum_sq = sum(L[i, k]² for k = 0 to i-1)
    ///   diag = A[i, i] - sum_sq
    ///
    ///   if diag <= 0:
    ///     return Error::NotPositiveDefinite
    ///
    ///   L[i, i] = sqrt(diag)
    ///
    ///   // Compute off-diagonal elements in column i
    ///   for j = i+1 to n-1:
    ///     sum_prod = sum(L[j, k] * L[i, k] for k = 0 to i-1)
    ///     L[j, i] = (A[j, i] - sum_prod) / L[i, i]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric positive-definite matrix [n, n]
    ///
    /// # Returns
    ///
    /// CholeskyDecomposition containing lower triangular factor L.
    ///
    /// # Errors
    ///
    /// Returns `Error::NotPositiveDefinite` if matrix is not positive-definite.
    fn cholesky_decompose(&self, a: &Tensor<R>) -> Result<CholeskyDecomposition<R>>;

    /// QR Decomposition using Householder reflections: A = QR
    ///
    /// # Algorithm (Householder)
    ///
    /// ```text
    /// Input: A [m, n] matrix
    /// Output: Q [m, m] orthogonal, R [m, n] upper triangular
    ///
    /// R = A.clone()
    /// Q = I_m  // Identity matrix
    ///
    /// For k = 0 to min(m, n) - 1:
    ///   // Extract column vector below diagonal
    ///   x = R[k:m, k]
    ///
    ///   // Compute Householder vector
    ///   alpha = -sign(x[0]) * ||x||
    ///   v = x.clone()
    ///   v[0] -= alpha
    ///   v = v / ||v||  // Normalize
    ///
    ///   // Apply reflection to R: R[k:m, k:n] -= 2 * v * (v^T @ R[k:m, k:n])
    ///   H_sub = I - 2 * v @ v^T
    ///   R[k:m, k:n] = H_sub @ R[k:m, k:n]
    ///
    ///   // Accumulate Q: Q = Q @ H (where H is full Householder matrix)
    ///   Q[:, k:m] = Q[:, k:m] @ H_sub
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix [m, n]
    ///
    /// # Returns
    ///
    /// QrDecomposition containing Q and R matrices.
    fn qr_decompose(&self, a: &Tensor<R>) -> Result<QrDecomposition<R>>;

    /// Thin QR Decomposition: A = QR where Q is [m, k] and R is [k, n], k = min(m, n)
    ///
    /// More memory efficient than full QR when m >> n.
    fn qr_decompose_thin(&self, a: &Tensor<R>) -> Result<QrDecomposition<R>>;

    // ========================================================================
    // Linear Solvers
    // ========================================================================

    /// Solve linear system Ax = b using LU decomposition
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Compute PA = LU
    /// 2. Apply permutation: Pb = b[pivots]
    /// 3. Forward substitution: Ly = Pb
    ///    for i = 0 to n-1:
    ///      y[i] = Pb[i] - sum(L[i,j] * y[j] for j = 0 to i-1)
    /// 4. Backward substitution: Ux = y
    ///    for i = n-1 down to 0:
    ///      x[i] = (y[i] - sum(U[i,j] * x[j] for j = i+1 to n-1)) / U[i,i]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Coefficient matrix [n, n]
    /// * `b` - Right-hand side [n] or [n, k] for multiple RHS
    ///
    /// # Returns
    ///
    /// Solution x with same shape as b.
    fn solve(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Solve triangular system Lx = b (forward substitution)
    ///
    /// # Algorithm
    ///
    /// ```text
    /// For i = 0 to n-1:
    ///   x[i] = (b[i] - sum(L[i,j] * x[j] for j = 0 to i-1)) / L[i,i]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `l` - Lower triangular matrix [n, n]
    /// * `b` - Right-hand side [n] or [n, k]
    /// * `unit_diagonal` - If true, assume L has unit diagonal (L[i,i] = 1)
    fn solve_triangular_lower(
        &self,
        l: &Tensor<R>,
        b: &Tensor<R>,
        unit_diagonal: bool,
    ) -> Result<Tensor<R>>;

    /// Solve triangular system Ux = b (backward substitution)
    ///
    /// # Algorithm
    ///
    /// ```text
    /// For i = n-1 down to 0:
    ///   x[i] = (b[i] - sum(U[i,j] * x[j] for j = i+1 to n-1)) / U[i,i]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `u` - Upper triangular matrix [n, n]
    /// * `b` - Right-hand side [n] or [n, k]
    fn solve_triangular_upper(&self, u: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Least squares solution: minimize ||Ax - b||²
    ///
    /// # Algorithm (QR-based)
    ///
    /// ```text
    /// 1. Compute A = QR (thin QR if m > n)
    /// 2. Compute Q^T @ b
    /// 3. Solve Rx = Q^T @ b (back substitution)
    /// ```
    ///
    /// For underdetermined systems (m < n), returns minimum-norm solution.
    ///
    /// # Arguments
    ///
    /// * `a` - Design matrix [m, n]
    /// * `b` - Target values [m] or [m, k]
    ///
    /// # Returns
    ///
    /// Solution x of shape [n] or [n, k].
    fn lstsq(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Matrix Operations
    // ========================================================================

    /// Matrix inverse using LU decomposition
    ///
    /// # Algorithm
    ///
    /// ```text
    /// A^(-1) = solve(A, I_n)
    ///
    /// More explicitly:
    /// 1. Compute PA = LU
    /// 2. For each column i of identity I_n:
    ///    Solve Ax_i = e_i (forward + backward substitution)
    /// 3. A^(-1) = [x_0, x_1, ..., x_{n-1}]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix [n, n]
    ///
    /// # Returns
    ///
    /// Inverse matrix A^(-1) [n, n].
    ///
    /// # Errors
    ///
    /// Returns `Error::SingularMatrix` if matrix is singular.
    fn inverse(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Matrix determinant using LU decomposition
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Compute PA = LU
    /// 2. det(A) = (-1)^num_swaps * product(U[i,i] for i = 0 to n-1)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix [n, n]
    ///
    /// # Returns
    ///
    /// Scalar determinant value (as 0-dimensional tensor).
    fn det(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Matrix trace: sum of diagonal elements
    ///
    /// # Algorithm
    ///
    /// ```text
    /// trace(A) = sum(A[i,i] for i = 0 to min(m,n)-1)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Matrix [m, n]
    ///
    /// # Returns
    ///
    /// Scalar trace value (as 0-dimensional tensor).
    fn trace(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Extract diagonal elements
    ///
    /// # Arguments
    ///
    /// * `a` - Matrix [m, n]
    ///
    /// # Returns
    ///
    /// 1D tensor of diagonal elements [min(m, n)].
    fn diag(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Create diagonal matrix from 1D tensor
    ///
    /// # Arguments
    ///
    /// * `a` - 1D tensor [n]
    ///
    /// # Returns
    ///
    /// Diagonal matrix [n, n] with a on diagonal, zeros elsewhere.
    fn diagflat(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Matrix rank via SVD
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Compute SVD: A = U @ S @ V^T
    /// 2. Count singular values > tolerance
    ///    tolerance = max(m, n) * eps * max(S)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Matrix [m, n]
    /// * `tol` - Optional tolerance (uses machine epsilon if None)
    ///
    /// # Returns
    ///
    /// Scalar rank value (as I64 tensor).
    fn matrix_rank(&self, a: &Tensor<R>, tol: Option<f64>) -> Result<Tensor<R>>;
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate matrix is 2D
pub fn validate_matrix_2d(shape: &[usize]) -> Result<(usize, usize)> {
    use crate::error::Error;

    if shape.len() != 2 {
        return Err(Error::Internal(format!(
            "Expected 2D matrix, got {}D tensor with shape {:?}",
            shape.len(),
            shape
        )));
    }
    Ok((shape[0], shape[1]))
}

/// Validate matrix is square
pub fn validate_square_matrix(shape: &[usize]) -> Result<usize> {
    let (m, n) = validate_matrix_2d(shape)?;
    if m != n {
        return Err(crate::error::Error::ShapeMismatch {
            expected: vec![m, m],
            got: vec![m, n],
        });
    }
    Ok(n)
}

/// Validate dtypes match for linear algebra operations
pub fn validate_linalg_dtype(dtype: DType) -> Result<()> {
    use crate::error::Error;

    match dtype {
        DType::F32 | DType::F64 => Ok(()),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "linear algebra",
        }),
    }
}

/// Machine epsilon for floating point comparison
pub fn machine_epsilon(dtype: DType) -> f64 {
    match dtype {
        DType::F32 => f32::EPSILON as f64,
        DType::F64 => f64::EPSILON,
        _ => f32::EPSILON as f64, // Default to F32 epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_square_matrix() {
        assert!(validate_square_matrix(&[3, 3]).is_ok());
        assert!(validate_square_matrix(&[2, 3]).is_err());
        assert!(validate_square_matrix(&[3, 2, 1]).is_err());
    }

    #[test]
    fn test_validate_linalg_dtype() {
        assert!(validate_linalg_dtype(DType::F32).is_ok());
        assert!(validate_linalg_dtype(DType::F64).is_ok());
        assert!(validate_linalg_dtype(DType::I32).is_err());
    }
}
