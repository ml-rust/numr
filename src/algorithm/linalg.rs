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

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

// ============================================================================
// Decomposition Results
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
    /// LU Decomposition with partial pivoting: PA = LU
    fn lu_decompose(&self, a: &Tensor<R>) -> Result<LuDecomposition<R>>;

    /// Cholesky Decomposition: A = LL^T
    fn cholesky_decompose(&self, a: &Tensor<R>) -> Result<CholeskyDecomposition<R>>;

    /// QR Decomposition using Householder reflections: A = QR
    fn qr_decompose(&self, a: &Tensor<R>) -> Result<QrDecomposition<R>>;

    /// Thin QR Decomposition: A = QR where Q is [m, k] and R is [k, n]
    fn qr_decompose_thin(&self, a: &Tensor<R>) -> Result<QrDecomposition<R>>;

    /// Solve linear system Ax = b using LU decomposition
    fn solve(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Solve triangular system Lx = b (forward substitution)
    fn solve_triangular_lower(
        &self,
        l: &Tensor<R>,
        b: &Tensor<R>,
        unit_diagonal: bool,
    ) -> Result<Tensor<R>>;

    /// Solve triangular system Ux = b (backward substitution)
    fn solve_triangular_upper(&self, u: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Least squares solution: minimize ||Ax - b||²
    fn lstsq(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Matrix inverse using LU decomposition
    fn inverse(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Matrix determinant using LU decomposition
    fn det(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Matrix trace: sum of diagonal elements
    fn trace(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Extract diagonal elements
    fn diag(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Create diagonal matrix from 1D tensor
    fn diagflat(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Matrix rank via SVD
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
        _ => f32::EPSILON as f64,
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
