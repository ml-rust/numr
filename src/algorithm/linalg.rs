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
// Matrix Norm Orders
// ============================================================================

/// Matrix norm order specification.
///
/// Different norms are appropriate for different use cases:
///
/// - **Frobenius**: General-purpose norm, similar to Euclidean distance.
///   Use for measuring overall matrix magnitude or computing loss functions.
///
/// - **Spectral**: Maximum amplification factor of the matrix.
///   Use for stability analysis, condition number estimation, or bounding
///   operator effects in neural networks.
///
/// - **Nuclear**: Sum of singular values (trace norm).
///   Use for matrix rank approximation, low-rank regularization, or
///   compressed sensing applications.
///
/// # Examples
///
/// ```ignore
/// use numr::algorithm::linalg::MatrixNormOrder;
///
/// // Frobenius norm: measures overall magnitude
/// // ||A||_F = sqrt(sum(A[i,j]²))
/// let fro_norm = client.matrix_norm(&matrix, MatrixNormOrder::Frobenius)?;
///
/// // Spectral norm: largest singular value (operator norm)
/// // ||A||_2 = sigma_max(A)
/// let spec_norm = client.matrix_norm(&matrix, MatrixNormOrder::Spectral)?;
///
/// // Nuclear norm: sum of singular values
/// // ||A||_* = sum(sigma_i)
/// let nuc_norm = client.matrix_norm(&matrix, MatrixNormOrder::Nuclear)?;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixNormOrder {
    /// Frobenius norm: sqrt(sum(A[i,j]²))
    ///
    /// The Frobenius norm treats the matrix as a flattened vector and computes
    /// its Euclidean length. It's always available since it only requires
    /// element-wise square, sum, and sqrt operations.
    Frobenius,

    /// Spectral norm (2-norm): maximum singular value
    ///
    /// The spectral norm equals the largest singular value of the matrix,
    /// which represents the maximum factor by which the matrix can stretch
    /// any input vector. Requires SVD computation.
    Spectral,

    /// Nuclear norm (trace norm): sum of singular values
    ///
    /// The nuclear norm equals the sum of all singular values. It's the
    /// tightest convex relaxation of matrix rank and is used in low-rank
    /// matrix recovery algorithms. Requires SVD computation.
    Nuclear,
}

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

    /// Matrix norm
    ///
    /// Computes the matrix norm of the input tensor.
    ///
    /// # Supported Norms
    ///
    /// - **Frobenius**: `sqrt(sum(A[i,j]²))` - Euclidean norm of the matrix
    /// - **Spectral** (2-norm): Maximum singular value (requires SVD)
    /// - **Nuclear** (trace norm): Sum of singular values (requires SVD)
    ///
    /// # Algorithm
    ///
    /// **Frobenius norm (currently implemented):**
    /// ```text
    /// ||A||_F = sqrt(sum_{i,j} |A[i,j]|^2) = sqrt(trace(A^T @ A))
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D matrix tensor
    /// * `ord` - Norm order: "fro" for Frobenius, "nuc" for nuclear (future), 2 for spectral (future)
    ///
    /// # Returns
    ///
    /// Scalar tensor containing the norm value
    fn matrix_norm(&self, a: &Tensor<R>, ord: MatrixNormOrder) -> Result<Tensor<R>>;

    /// Singular Value Decomposition: A = U @ diag(S) @ V^T
    ///
    /// Computes the thin SVD using the One-Sided Jacobi algorithm.
    /// This algorithm is chosen because it produces identical results
    /// across all backends (CPU, CUDA, WebGPU), which is required for
    /// numerical parity.
    ///
    /// # Algorithm: One-Sided Jacobi SVD
    ///
    /// ```text
    /// Input: A [m, n] where we compute thin SVD
    /// Output: U [m, k], S [k], V^T [k, n] where k = min(m, n)
    ///
    /// 1. If m < n: Transpose A, compute SVD, swap U↔V^T
    /// 2. Initialize: B = A (working copy), V = I_n
    /// 3. REPEAT (max 30 sweeps):
    ///    FOR each pair (p, q) where p < q:
    ///      - Compute Gram elements: a_pp, a_qq, a_pq = B[:,p]·B[:,q]
    ///      - If |a_pq| > tol: compute Jacobi rotation (c,s), apply to B and V columns
    ///    Check convergence: sqrt(Σ a_pq²) < n * epsilon
    /// 4. Extract: S[j] = ||B[:,j]||, U[:,j] = B[:,j]/S[j]
    /// 5. Sort S descending, reorder U and V columns accordingly
    /// 6. Return U, S, V^T = V.transpose()
    /// ```
    ///
    /// # Numerical Stability
    ///
    /// - Jacobi rotations are computed using the stable formula from LAPACK
    /// - Convergence tolerance: n * machine_epsilon (1e-7 for F32, 1e-15 for F64)
    /// - Maximum iterations: 30 sweeps (typically converges in 5-10)
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D matrix tensor [m, n]
    ///
    /// # Returns
    ///
    /// `SvdDecomposition` containing:
    /// - `u`: Left singular vectors [m, k]
    /// - `s`: Singular values [k] (sorted in descending order)
    /// - `vt`: Right singular vectors transposed [k, n]
    ///
    /// # Errors
    ///
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    /// - `Internal` if convergence is not reached in max iterations
    ///
    /// # Edge Cases
    ///
    /// - **Empty matrices** (m=0 or n=0): Returns tensors with correct shapes
    ///   (U: [m, 0], S: [0], V^T: [0, n]) but no computation is performed.
    ///
    /// - **Single element** (1×1 matrix): Returns U=[1], S=[|a|], V^T=[sign(a)]
    ///   where sign(a) = 1 for a≥0, -1 for a<0.
    ///
    /// - **Rank-deficient matrices**: Zero or near-zero singular values are
    ///   included in output. The number of non-zero singular values indicates
    ///   the numerical rank. Use a tolerance (e.g., S > 1e-10) to determine rank.
    ///
    /// - **Very thin matrices** (m >> n or m << n): Works correctly via transpose.
    ///   Performance is O(min(m,n)² × max(m,n)) regardless of orientation.
    ///
    /// - **Numerical precision**: For ill-conditioned matrices (high condition
    ///   number), use F64 instead of F32. F32 may lose precision for matrices
    ///   with singular value ratios > 10^6.
    ///
    /// - **Non-convergence**: Extremely rare with Jacobi method. If it occurs,
    ///   an `Internal` error is returned. The matrix may be numerically
    ///   pathological (e.g., contains NaN/Inf).
    ///
    /// # Backend Notes
    ///
    /// - **CPU/CUDA**: Supports F32 and F64
    /// - **WebGPU**: F32 only (WGSL limitation)
    /// - All backends use identical One-Sided Jacobi algorithm for numerical parity
    ///
    /// # Implementation Details (Sorting)
    ///
    /// The core Jacobi algorithm is identical across all backends. Post-processing
    /// (sorting singular values and reordering U/V) may differ in execution location:
    /// - **WebGPU**: Sorting done in WGSL shader (simple selection sort, GPU-native)
    /// - **CPU/CUDA**: Sorting done in Rust (std::sort, after kernel execution)
    ///
    /// Both approaches produce identical results because:
    /// 1. Sorting algorithm is deterministic (same input → same output)
    /// 2. Only the sorted order matters, not the sorting algorithm itself
    /// 3. Final U, S, V^T satisfy A = U @ diag(S) @ V^T with S[i] ≥ S[i+1]
    fn svd_decompose(&self, a: &Tensor<R>) -> Result<SvdDecomposition<R>>;
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
