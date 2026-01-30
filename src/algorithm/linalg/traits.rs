//! Linear algebra algorithm trait definitions
//!
//! These traits define the contract that all backends must implement
//! to ensure numerical parity across CPU, CUDA, WebGPU, and other backends.

use super::MatrixNormOrder;
use super::decompositions::*;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Trait for matrix function operations (expm, logm, sqrtm)
///
/// Matrix functions extend scalar functions to matrices. For a scalar function f
/// and a diagonalizable matrix A = V @ diag(λ) @ V^{-1}, the matrix function is:
///
/// f(A) = V @ diag(f(λ)) @ V^{-1}
///
/// For non-diagonalizable matrices, the Schur decomposition is used:
/// A = Z @ T @ Z^T, then f(A) = Z @ f(T) @ Z^T
///
/// where f(T) is computed using special formulas for quasi-triangular matrices.
///
/// # Implemented Functions
///
/// - **expm**: Matrix exponential e^A
/// - **logm**: Matrix logarithm (principal branch)
/// - **sqrtm**: Matrix square root (principal branch)
///
/// # Use Cases
///
/// - **expm**: Solving linear ODEs dx/dt = Ax → x(t) = e^{At} x(0)
/// - **logm**: Computing matrix powers A^p = e^{p*log(A)}
/// - **sqrtm**: Polar decomposition, control theory
pub trait MatrixFunctionsAlgorithms<R: Runtime> {
    /// Matrix exponential: e^A
    ///
    /// Computes the matrix exponential using the Schur-Parlett algorithm:
    /// 1. Compute Schur decomposition: A = Z @ T @ Z^T
    /// 2. Compute exp(T) for quasi-triangular T
    /// 3. Reconstruct: exp(A) = Z @ exp(T) @ Z^T
    ///
    /// # Algorithm for Quasi-Triangular T
    ///
    /// For 1×1 diagonal blocks: exp(t_ii) = e^{t_ii}
    ///
    /// For 2×2 diagonal blocks (complex conjugate eigenvalues a ± bi):
    /// ```text
    /// exp([a, b; -b, a]) = e^a * [cos(b), sin(b); -sin(b), cos(b)]
    /// ```
    ///
    /// For off-diagonal elements, use the formula:
    /// ```text
    /// exp(T)[i,j] = (exp(T[i,i]) - exp(T[j,j])) / (T[i,i] - T[j,j]) * T[i,j]
    ///              when T[i,i] ≠ T[j,j]
    /// exp(T)[i,j] = exp(T[i,i]) * T[i,j]  when T[i,i] = T[j,j]
    /// ```
    ///
    /// # Properties
    ///
    /// - exp(0) = I (identity)
    /// - exp(A + B) = exp(A) @ exp(B) if AB = BA
    /// - det(exp(A)) = e^{tr(A)}
    /// - exp(A)^{-1} = exp(-A)
    fn expm(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Matrix logarithm: log(A) (principal branch)
    ///
    /// Computes the principal matrix logarithm using Schur decomposition
    /// with inverse scaling and squaring.
    ///
    /// # Requirements
    ///
    /// The matrix A must have no eigenvalues on the closed negative real axis.
    fn logm(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Matrix square root: A^{1/2} (principal branch)
    ///
    /// Computes the principal square root using Denman-Beavers iteration:
    /// ```text
    /// Y_0 = A, Z_0 = I
    /// REPEAT:
    ///   Y_{k+1} = (Y_k + Z_k^{-1}) / 2
    ///   Z_{k+1} = (Z_k + Y_k^{-1}) / 2
    /// UNTIL convergence
    /// sqrt(A) = Y_∞
    /// ```
    ///
    /// # Requirements
    ///
    /// The matrix A must have no eigenvalues on the closed negative real axis.
    fn sqrtm(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Matrix sign function: sign(A)
    ///
    /// Computes the matrix sign function using Newton iteration:
    /// ```text
    /// X_0 = A
    /// REPEAT:
    ///   X_{k+1} = (X_k + X_k^{-1}) / 2
    /// UNTIL convergence
    /// sign(A) = X_∞
    /// ```
    ///
    /// # Properties
    ///
    /// - sign(A)^2 = I (involutory)
    /// - Eigenvalues of sign(A) are +1 or -1
    fn signm(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Fractional matrix power: A^p for any real p
    ///
    /// Computes A^p using: A^p = exp(p * log(A))
    ///
    /// # Special Cases
    ///
    /// - p = 0: Returns identity matrix
    /// - p = 1: Returns A unchanged
    /// - p = -1: Returns matrix inverse
    /// - p = 0.5: Equivalent to sqrtm(A)
    /// - Integer p: Uses repeated squaring
    fn fractional_matrix_power(&self, a: &Tensor<R>, p: f64) -> Result<Tensor<R>>;

    /// General matrix function: f(A) for any scalar function f
    ///
    /// Computes f(A) using the Schur-Parlett algorithm.
    ///
    /// # Closure Requirements
    ///
    /// The closure `f` must be `Send + Sync` to support GPU backends (CUDA, WebGPU).
    /// This allows the function to be safely captured and potentially executed across
    /// GPU threads or transferred to device memory contexts.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Custom matrix function: f(x) = sin(x)
    /// let result = client.funm(&matrix, |x| x.sin())?;
    /// ```
    fn funm<F>(&self, a: &Tensor<R>, f: F) -> Result<Tensor<R>>
    where
        F: Fn(f64) -> f64 + Send + Sync;
}

/// Algorithmic contract for linear algebra operations
///
/// All backends implementing linear algebra MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
///
/// # Implementation Requirements
///
/// Backends may differ in:
/// - Parallelization strategy (threads, SIMD, GPU blocks)
/// - Memory access patterns (blocking, tiling)
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

    /// Matrix norm (Frobenius, Spectral, or Nuclear)
    fn matrix_norm(&self, a: &Tensor<R>, ord: MatrixNormOrder) -> Result<Tensor<R>>;

    /// Singular Value Decomposition: A = U @ diag(S) @ V^T
    fn svd_decompose(&self, a: &Tensor<R>) -> Result<SvdDecomposition<R>>;

    /// Eigendecomposition for symmetric matrices: A = V @ diag(λ) @ V^T
    fn eig_decompose_symmetric(&self, a: &Tensor<R>) -> Result<EigenDecomposition<R>>;

    /// General Eigendecomposition for non-symmetric matrices
    fn eig_decompose(&self, a: &Tensor<R>) -> Result<GeneralEigenDecomposition<R>>;

    /// Schur Decomposition: A = Z @ T @ Z^T
    fn schur_decompose(&self, a: &Tensor<R>) -> Result<SchurDecomposition<R>>;

    /// Moore-Penrose pseudo-inverse via SVD
    fn pinverse(&self, a: &Tensor<R>, rcond: Option<f64>) -> Result<Tensor<R>>;

    /// Matrix condition number via SVD
    fn cond(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Covariance matrix
    fn cov(&self, a: &Tensor<R>, ddof: Option<usize>) -> Result<Tensor<R>>;

    /// Correlation coefficient matrix (Pearson correlation)
    fn corrcoef(&self, a: &Tensor<R>) -> Result<Tensor<R>>;
}
