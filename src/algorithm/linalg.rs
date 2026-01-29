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

/// Eigendecomposition result for symmetric matrices: A = V @ diag(λ) @ V^T
///
/// For a real symmetric matrix A, all eigenvalues are real and eigenvectors
/// form an orthonormal basis. The decomposition satisfies:
/// - A @ V[:,i] = λ[i] * V[:,i] for each eigenpair
/// - V^T @ V = I (eigenvectors are orthonormal)
/// - A = V @ diag(λ) @ V^T
pub struct EigenDecomposition<R: Runtime> {
    /// Eigenvalues λ [n] (sorted in descending order by magnitude)
    pub eigenvalues: Tensor<R>,

    /// Eigenvector matrix V [n, n] where V[:,i] is the eigenvector for λ[i]
    /// Columns are orthonormal: V^T @ V = I
    pub eigenvectors: Tensor<R>,
}

/// Schur decomposition result: A = Z @ T @ Z^T
///
/// For a real square matrix A, the Schur decomposition factors A into:
/// - Z: orthogonal matrix (Z^T @ Z = I)
/// - T: upper quasi-triangular (real Schur form)
///
/// The real Schur form T has:
/// - Real eigenvalues on the diagonal (1×1 blocks)
/// - Complex conjugate eigenvalue pairs in 2×2 blocks on the diagonal
///
/// # Properties
///
/// - A = Z @ T @ Z^T (reconstruction)
/// - Z^T @ Z = I (orthogonality)
/// - Eigenvalues of A are the diagonal blocks of T
/// - For symmetric A, T is diagonal (eigenvalue decomposition)
///
/// # Use Cases
///
/// - Matrix function computation (expm, logm, sqrtm)
/// - Solving matrix equations (Sylvester, Lyapunov)
/// - Stability analysis of dynamical systems
pub struct SchurDecomposition<R: Runtime> {
    /// Orthogonal transformation matrix Z [n, n]
    /// Columns form an orthonormal basis: Z^T @ Z = I
    pub z: Tensor<R>,

    /// Upper quasi-triangular Schur form T [n, n]
    /// Contains 1×1 blocks for real eigenvalues and
    /// 2×2 blocks for complex conjugate eigenvalue pairs
    pub t: Tensor<R>,
}

/// General eigendecomposition result for non-symmetric matrices: A @ V = V @ diag(λ)
///
/// For a real non-symmetric matrix A, eigenvalues may be complex even though
/// A is real. Complex eigenvalues always come in conjugate pairs.
///
/// # Eigenvalue Storage
///
/// Eigenvalues are stored as separate real and imaginary parts:
/// - `eigenvalues_real[i]` = Re(λ_i)
/// - `eigenvalues_imag[i]` = Im(λ_i)
///
/// For real eigenvalues, `eigenvalues_imag[i] = 0`.
/// For complex conjugate pairs, they appear consecutively:
/// - λ_j = a + bi (eigenvalues_real[j] = a, eigenvalues_imag[j] = b)
/// - λ_{j+1} = a - bi (eigenvalues_real[j+1] = a, eigenvalues_imag[j+1] = -b)
///
/// # Eigenvector Storage
///
/// Eigenvectors are stored as separate real and imaginary matrices.
/// For a real eigenvalue at index j, the eigenvector is:
/// - `eigenvectors_real[:, j]` (imaginary part is zero)
///
/// For complex conjugate pairs at indices j and j+1:
/// - Eigenvector for λ_j = a + bi is: `eigenvectors_real[:, j] + i * eigenvectors_imag[:, j]`
/// - Eigenvector for λ_{j+1} = a - bi is: `eigenvectors_real[:, j] - i * eigenvectors_imag[:, j]`
///
/// Note: For complex pairs, columns j and j+1 of eigenvectors_real contain the same data,
/// and columns j and j+1 of eigenvectors_imag contain negatives of each other.
///
/// # Properties
///
/// - A @ v_i = λ_i * v_i for each eigenpair
/// - Complex eigenvalues come in conjugate pairs (for real A)
/// - Eigenvectors may not be orthogonal (unlike symmetric case)
///
/// # Use Cases
///
/// - Stability analysis of dynamical systems (eigenvalues with positive real parts → unstable)
/// - Matrix exponential computation
/// - Solving systems of linear ODEs
pub struct GeneralEigenDecomposition<R: Runtime> {
    /// Real parts of eigenvalues [n]
    pub eigenvalues_real: Tensor<R>,

    /// Imaginary parts of eigenvalues [n]
    /// Zero for real eigenvalues, non-zero for complex conjugate pairs
    pub eigenvalues_imag: Tensor<R>,

    /// Real parts of eigenvector matrix [n, n]
    /// Column j is the real part of eigenvector for λ_j
    pub eigenvectors_real: Tensor<R>,

    /// Imaginary parts of eigenvector matrix [n, n]
    /// Column j is the imaginary part of eigenvector for λ_j
    /// Zero for real eigenvalues
    pub eigenvectors_imag: Tensor<R>,
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
///
/// # Backend-Specific DType Support
///
/// | Backend | Supported DTypes | Notes |
/// |---------|-----------------|-------|
/// | CPU     | F32, F64        | Full precision support |
/// | CUDA    | F32, F64        | Full precision support |
/// | WebGPU  | F32 only        | WGSL language limitation |
///
/// WebGPU users requiring F64 precision should use CPU or CUDA backends instead.
/// Some methods (like `lstsq`) may have additional WebGPU-specific limitations
/// documented on the individual methods.
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
    ///
    /// Computes the solution x that minimizes the 2-norm of the residual ||Ax - b||².
    /// Uses QR decomposition (Householder reflections) followed by back-substitution.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Compute QR decomposition: A = QR
    /// 2. Transform: y = Q^T @ b
    /// 3. Solve: R @ x = y (back-substitution)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Coefficient matrix [m, n]
    /// * `b` - Right-hand side vector/matrix [m] or [m, k]
    ///
    /// # Returns
    ///
    /// Solution tensor x [n] or [n, k]
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if dimensions are incompatible
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Backend Notes
    ///
    /// - **CPU/CUDA**: Supports F32/F64 and both 1D vectors and 2D matrices for b
    /// - **WebGPU**: F32 only, and b must be 1D (single right-hand side vector)
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

    /// Eigendecomposition for symmetric matrices: A = V @ diag(λ) @ V^T
    ///
    /// Computes eigenvalues and eigenvectors of a real symmetric matrix using
    /// the Jacobi eigenvalue algorithm. This algorithm is chosen because it
    /// produces identical results across all backends (CPU, CUDA, WebGPU),
    /// which is required for numerical parity.
    ///
    /// # Algorithm: Jacobi Eigenvalue Algorithm
    ///
    /// ```text
    /// Input: A [n, n] symmetric matrix
    /// Output: eigenvalues [n], eigenvectors [n, n]
    ///
    /// 1. Initialize: V = I_n (eigenvector matrix starts as identity)
    /// 2. REPEAT (max 30 sweeps):
    ///    FOR each pair (p, q) where p < q:
    ///      - If |A[p,q]| > tol:
    ///        a. Compute Jacobi rotation angle θ from A[p,p], A[q,q], A[p,q]
    ///        b. Apply rotation: A' = J^T @ A @ J (zeros out A[p,q] and A[q,p])
    ///        c. Update eigenvectors: V = V @ J
    ///    Check convergence: max(|A[i,j]| for i≠j) < n * epsilon
    /// 3. eigenvalues = diag(A) (diagonal elements after convergence)
    /// 4. Sort eigenvalues descending by magnitude, reorder eigenvector columns
    /// 5. Return eigenvalues, eigenvectors
    /// ```
    ///
    /// # Jacobi Rotation
    ///
    /// For off-diagonal element A[p,q], the rotation angle θ is computed as:
    /// ```text
    /// τ = (A[q,q] - A[p,p]) / (2 * A[p,q])
    /// t = sign(τ) / (|τ| + sqrt(1 + τ²))   [stable formula]
    /// c = 1 / sqrt(1 + t²)
    /// s = t * c
    /// ```
    ///
    /// The rotation matrix J is identity except:
    /// ```text
    /// J[p,p] = c,  J[q,q] = c
    /// J[p,q] = s,  J[q,p] = -s
    /// ```
    ///
    /// # Properties of Output
    ///
    /// - **Eigenvalues are real**: Guaranteed for symmetric matrices
    /// - **Eigenvectors are orthonormal**: V^T @ V = V @ V^T = I
    /// - **Reconstruction**: A = V @ diag(λ) @ V^T (within tolerance)
    /// - **Sorted by magnitude**: |λ[0]| ≥ |λ[1]| ≥ ... ≥ |λ[n-1]|
    ///
    /// # Numerical Stability
    ///
    /// - Jacobi rotations use stable formulas from LAPACK
    /// - Convergence tolerance: n * machine_epsilon (1e-7 for F32, 1e-15 for F64)
    /// - Maximum iterations: 30 sweeps (typically converges in 5-10)
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D symmetric matrix tensor [n, n]
    ///
    /// # Returns
    ///
    /// `EigenDecomposition` containing:
    /// - `eigenvalues`: Real eigenvalues [n] (sorted by magnitude, descending)
    /// - `eigenvectors`: Orthonormal eigenvector matrix [n, n]
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if input is not square
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    /// - `Internal` if convergence is not reached in max iterations
    ///
    /// # Edge Cases
    ///
    /// - **1×1 matrix**: Returns eigenvalue = A[0,0], eigenvector = [1.0]
    /// - **Diagonal matrix**: Returns diagonal elements as eigenvalues, identity as eigenvectors
    /// - **Zero matrix**: Returns all zero eigenvalues, identity eigenvectors
    /// - **Degenerate eigenvalues**: Multiple eigenvalues with same value are handled;
    ///   corresponding eigenvectors span the eigenspace but may not be unique
    ///
    /// # Note on Symmetry
    ///
    /// This function assumes the input matrix is symmetric (A = A^T).
    /// Only the lower triangular part is read; the upper triangular part is ignored.
    /// For non-symmetric matrices, use SVD instead.
    ///
    /// # Backend Notes
    ///
    /// - **CPU/CUDA**: Supports F32 and F64
    /// - **WebGPU**: F32 only (WGSL limitation)
    /// - All backends use identical Jacobi eigenvalue algorithm for numerical parity
    fn eig_decompose_symmetric(&self, a: &Tensor<R>) -> Result<EigenDecomposition<R>>;

    /// General Eigendecomposition for non-symmetric matrices: A @ V = V @ diag(λ)
    ///
    /// Computes eigenvalues and eigenvectors of a general (non-symmetric) real matrix.
    /// Unlike symmetric matrices, eigenvalues may be complex even for real input matrices.
    /// Complex eigenvalues always come in conjugate pairs for real matrices.
    ///
    /// # Algorithm: Schur Decomposition + Back-Substitution
    ///
    /// ```text
    /// Input: A [n, n] square matrix
    /// Output: eigenvalues (real + imag), eigenvectors (real + imag)
    ///
    /// 1. Compute Schur decomposition: A = Z @ T @ Z^T
    /// 2. Extract eigenvalues from T:
    ///    - 1×1 diagonal blocks → real eigenvalue
    ///    - 2×2 diagonal blocks → complex conjugate pair
    /// 3. Compute eigenvectors by back-substitution on (T - λI)
    /// 4. Transform eigenvectors back: V = Z @ V_T
    /// ```
    ///
    /// # Eigenvalue Extraction from Schur Form
    ///
    /// The quasi-triangular Schur form T has eigenvalues on the diagonal:
    /// - Real eigenvalue at position (i, i): λ = T[i, i]
    /// - Complex pair at positions (i, i) to (i+1, i+1):
    ///   ```text
    ///   [ a  b ]
    ///   [ c  a ]  → λ = a ± i*sqrt(-b*c) for b*c < 0
    ///   ```
    ///
    /// # Output Format
    ///
    /// Returns `GeneralEigenDecomposition` with separate real/imaginary parts:
    /// - `eigenvalues_real[i]`, `eigenvalues_imag[i]` for λ_i
    /// - `eigenvectors_real[:, i]`, `eigenvectors_imag[:, i]` for v_i
    ///
    /// For complex conjugate pairs at indices j and j+1:
    /// - λ_j = a + bi, λ_{j+1} = a - bi
    /// - v_j = u + iw, v_{j+1} = u - iw (stored in columns j, j+1)
    ///
    /// # Numerical Stability
    ///
    /// - Uses Schur decomposition (backward stable)
    /// - Eigenvector computation uses column-oriented back-substitution
    /// - For defective matrices (repeated eigenvalues with insufficient eigenvectors),
    ///   the algorithm returns generalized eigenvectors
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D square matrix tensor [n, n]
    ///
    /// # Returns
    ///
    /// `GeneralEigenDecomposition` containing:
    /// - `eigenvalues_real`: Real parts of eigenvalues [n]
    /// - `eigenvalues_imag`: Imaginary parts of eigenvalues [n]
    /// - `eigenvectors_real`: Real parts of eigenvectors [n, n]
    /// - `eigenvectors_imag`: Imaginary parts of eigenvectors [n, n]
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if input is not square
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    /// - `Internal` if Schur decomposition fails to converge
    ///
    /// # Edge Cases
    ///
    /// - **1×1 matrix**: Returns eigenvalue = A[0,0], eigenvector = [1.0]
    /// - **Symmetric matrix**: All eigenvalues real, equivalent to `eig_decompose_symmetric`
    /// - **Defective matrix**: Returns generalized eigenvectors for repeated eigenvalues
    /// - **Upper triangular**: Eigenvalues are diagonal elements
    ///
    /// # Example (Conceptual)
    ///
    /// ```ignore
    /// // Matrix with complex eigenvalues
    /// let a = Tensor::from_slice(&[0.0, -1.0, 1.0, 0.0], &[2, 2], &client)?;
    /// let eig = client.eig_decompose(&a)?;
    /// // eigenvalues: 0 ± 1i
    /// // eig.eigenvalues_real = [0.0, 0.0]
    /// // eig.eigenvalues_imag = [1.0, -1.0]
    /// ```
    ///
    /// # Backend Notes
    ///
    /// - **CPU/CUDA**: Supports F32 and F64
    /// - **WebGPU**: F32 only (WGSL limitation)
    /// - All backends use identical algorithm for numerical parity
    fn eig_decompose(&self, a: &Tensor<R>) -> Result<GeneralEigenDecomposition<R>>;

    /// Schur Decomposition: A = Z @ T @ Z^T
    ///
    /// Computes the Schur decomposition of a real square matrix. The decomposition
    /// factors the matrix A into an orthogonal matrix Z and an upper quasi-triangular
    /// matrix T (the real Schur form).
    ///
    /// # Algorithm: QR Iteration with Shifts
    ///
    /// ```text
    /// Input: A [n, n] square matrix
    /// Output: Z [n, n] orthogonal, T [n, n] quasi-triangular
    ///
    /// 1. Reduce to Hessenberg form: H = Q0^T @ A @ Q0
    /// 2. QR iteration with implicit double shifts:
    ///    REPEAT until convergence:
    ///      a. Compute shift (Wilkinson or Francis)
    ///      b. QR factorize: H - μI = Q_k @ R_k
    ///      c. Update: H = R_k @ Q_k + μI
    ///      d. Accumulate: Z = Z @ Q_k
    ///    Check: subdiagonal elements < tol * (|A[i,i]| + |A[i+1,i+1]|)
    /// 3. T = final H, Z = accumulated transformations
    /// ```
    ///
    /// The algorithm uses Francis QR double-shift for efficiency, which
    /// handles both real and complex conjugate eigenvalue pairs without
    /// explicitly using complex arithmetic.
    ///
    /// # Real Schur Form (Quasi-Triangular)
    ///
    /// The output T is quasi-triangular:
    /// - Real eigenvalues appear as 1×1 diagonal blocks
    /// - Complex conjugate pairs appear as 2×2 diagonal blocks:
    ///   ```text
    ///   [ a  b ]
    ///   [-c  a ]  where eigenvalues are a ± i*sqrt(b*c)
    ///   ```
    ///
    /// # Numerical Stability
    ///
    /// - Initial Hessenberg reduction preserves eigenvalues exactly
    /// - QR iteration is backward stable
    /// - Convergence tolerance: n * machine_epsilon
    /// - Maximum iterations: 30 * n (rarely needed)
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D square matrix tensor [n, n]
    ///
    /// # Returns
    ///
    /// `SchurDecomposition` containing:
    /// - `z`: Orthogonal matrix [n, n] where Z^T @ Z = I
    /// - `t`: Upper quasi-triangular (real Schur form) [n, n]
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if input is not square
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    /// - `Internal` if convergence is not reached in max iterations
    ///
    /// # Edge Cases
    ///
    /// - **1×1 matrix**: Returns T = A, Z = [1]
    /// - **2×2 matrix**: Computed directly from characteristic polynomial
    /// - **Upper triangular**: Returns T = A, Z = I (already in Schur form)
    /// - **Symmetric matrix**: T is diagonal (eigenvalues), Z = eigenvectors
    ///
    /// # Backend Notes
    ///
    /// - **CPU/CUDA**: Supports F32 and F64
    /// - **WebGPU**: F32 only (WGSL limitation)
    /// - All backends use identical QR iteration algorithm for numerical parity
    fn schur_decompose(&self, a: &Tensor<R>) -> Result<SchurDecomposition<R>>;

    /// Moore-Penrose pseudo-inverse via SVD: A^+ = V @ diag(1/S) @ U^T
    ///
    /// Computes the pseudo-inverse of a matrix using SVD. For a matrix A with
    /// SVD decomposition A = U @ diag(S) @ V^T, the pseudo-inverse is:
    ///
    /// ```text
    /// A^+ = V @ diag(1/S_i where S_i > tol, else 0) @ U^T
    /// ```
    ///
    /// # Algorithm
    ///
    /// 1. Compute SVD: A = U @ diag(S) @ V^T
    /// 2. Invert non-zero singular values: S_inv[i] = 1/S[i] if S[i] > tol, else 0
    /// 3. Compute: A^+ = V @ diag(S_inv) @ U^T
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D matrix tensor [m, n]
    /// * `rcond` - Cutoff ratio for small singular values. Singular values less than
    ///             rcond * max(S) are treated as zero. Default: n * epsilon
    ///
    /// # Returns
    ///
    /// Pseudo-inverse matrix [n, m]
    ///
    /// # Properties
    ///
    /// The pseudo-inverse satisfies the Moore-Penrose conditions:
    /// - A @ A^+ @ A = A
    /// - A^+ @ A @ A^+ = A^+
    /// - (A @ A^+)^T = A @ A^+
    /// - (A^+ @ A)^T = A^+ @ A
    ///
    /// # Edge Cases
    ///
    /// - **Empty matrix** (m=0 or n=0): Returns valid tensor with shape [n, m] and 0 elements
    /// - **Rank-deficient matrix**: Small singular values below rcond * max(S) are treated as zero
    /// - **Ill-conditioned matrix**: Works correctly but may have numerical error in result
    ///
    /// # Errors
    ///
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Backend Notes
    ///
    /// - **CPU/CUDA**: Supports F32 and F64
    /// - **WebGPU**: F32 only (WGSL limitation)
    fn pinverse(&self, a: &Tensor<R>, rcond: Option<f64>) -> Result<Tensor<R>>;

    /// Matrix condition number via SVD: cond(A) = σ_max / σ_min
    ///
    /// The condition number measures how sensitive the solution of a linear system
    /// Ax = b is to perturbations in A and b. A high condition number indicates
    /// an ill-conditioned matrix.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Compute SVD: A = U @ diag(S) @ V^T
    /// 2. cond(A) = max(S) / min(S) where S > 0
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D matrix tensor [m, n]
    ///
    /// # Returns
    ///
    /// Scalar tensor containing the condition number
    ///
    /// # Edge Cases
    ///
    /// - **Empty matrix** (m=0 or n=0): Returns infinity (scalar tensor)
    /// - **Singular matrix** (min(S) ≈ 0): Returns infinity
    /// - **Identity matrix**: cond(I) = 1.0
    /// - **1x1 matrix**: cond([a]) = 1.0 (always well-conditioned)
    ///
    /// # Errors
    ///
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Backend Notes
    ///
    /// - **CPU/CUDA**: Supports F32 and F64
    /// - **WebGPU**: F32 only (WGSL limitation)
    fn cond(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Covariance matrix of a set of observations
    ///
    /// Computes the covariance matrix from a matrix where each row is an
    /// observation and each column is a variable (feature).
    ///
    /// # Algorithm
    ///
    /// For input X [n_samples, n_features]:
    ///
    /// ```text
    /// 1. Center data: X_centered = X - mean(X, axis=0)
    /// 2. Compute covariance: cov = X_centered^T @ X_centered / (n_samples - ddof)
    /// ```
    ///
    /// Where ddof (delta degrees of freedom) is:
    /// - 0 for population covariance (biased)
    /// - 1 for sample covariance (unbiased, default)
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D matrix tensor [n_samples, n_features]
    /// * `ddof` - Delta degrees of freedom. Default: 1 (sample covariance)
    ///
    /// # Returns
    ///
    /// Covariance matrix [n_features, n_features]
    ///
    /// # Properties
    ///
    /// - The covariance matrix is symmetric: cov[i,j] = cov[j,i]
    /// - Diagonal elements are variances: cov[i,i] = var(X[:,i])
    /// - Off-diagonal elements are covariances: cov[i,j] = cov(X[:,i], X[:,j])
    ///
    /// # Edge Cases
    ///
    /// - **Zero features** (n_features=0): Returns empty matrix [0, 0]
    /// - **Single feature**: Returns 1x1 matrix with variance
    /// - **Constant feature** (zero variance): Corresponding diagonal element is 0
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if input is not 2D
    /// - `Internal` if n_samples <= ddof (not enough samples for specified ddof)
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Backend Notes
    ///
    /// - **CPU/CUDA**: Supports F32 and F64
    /// - **WebGPU**: F32 only (WGSL limitation)
    fn cov(&self, a: &Tensor<R>, ddof: Option<usize>) -> Result<Tensor<R>>;

    /// Correlation coefficient matrix (Pearson correlation)
    ///
    /// Computes the correlation coefficient matrix from a matrix where each row is
    /// an observation and each column is a variable (feature).
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Compute covariance matrix: C = cov(X)
    /// 2. Extract standard deviations: std[i] = sqrt(C[i,i])
    /// 3. Normalize: corr[i,j] = C[i,j] / (std[i] * std[j])
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D matrix tensor [n_samples, n_features]
    ///
    /// # Returns
    ///
    /// Correlation matrix [n_features, n_features]
    ///
    /// # Properties
    ///
    /// - Diagonal elements are 1.0 for non-constant variables: corr[i,i] = 1.0
    /// - Off-diagonal elements are in [-1, 1]: -1 ≤ corr[i,j] ≤ 1
    /// - The matrix is symmetric: corr[i,j] = corr[j,i]
    ///
    /// # Edge Cases
    ///
    /// - **Zero features** (n_features=0): Returns empty matrix [0, 0]
    /// - **Zero variance** (constant variable):
    ///   - Diagonal: corr[i,i] = 0.0 (not 1.0, since variance is 0)
    ///   - Off-diagonal: corr[i,j] = 0.0 (undefined correlation set to 0)
    ///   - This differs from some implementations that return NaN
    /// - **Perfect correlation**: corr[i,j] = ±1.0 (clamped to handle numerical error)
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if input is not 2D
    /// - `Internal` if n_samples <= 1 (need at least 2 samples for correlation)
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Backend Notes
    ///
    /// - **CPU/CUDA**: Supports F32 and F64
    /// - **WebGPU**: F32 only (WGSL limitation)
    fn corrcoef(&self, a: &Tensor<R>) -> Result<Tensor<R>>;
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
