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

    /// Kronecker product: A ⊗ B
    ///
    /// Computes the Kronecker product of two matrices. For A of shape [m, n]
    /// and B of shape [p, q], the result has shape [m*p, n*q].
    ///
    /// # Definition
    ///
    /// ```text
    /// (A ⊗ B)[i*p + k, j*q + l] = A[i, j] * B[k, l]
    /// ```
    ///
    /// Equivalently, the Kronecker product replaces each element a_ij of A
    /// with the block a_ij * B.
    ///
    /// # Properties
    ///
    /// - (A ⊗ B) ⊗ C = A ⊗ (B ⊗ C) (associative)
    /// - A ⊗ (B + C) = A ⊗ B + A ⊗ C (distributive)
    /// - (A ⊗ B)^T = A^T ⊗ B^T
    /// - (A ⊗ B)(C ⊗ D) = (AC) ⊗ (BD) (mixed-product property)
    /// - det(A ⊗ B) = det(A)^q * det(B)^m for square matrices
    ///
    /// # Use Cases
    ///
    /// - Quantum computing (tensor products of quantum states)
    /// - Control theory (Sylvester/Lyapunov equation solvers)
    /// - Signal processing (2D filtering)
    /// - Graph theory (graph products)
    fn kron(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Khatri-Rao product (column-wise Kronecker product): A ⊙ B
    ///
    /// Computes the column-wise Kronecker product of two matrices.
    /// For A of shape [m, k] and B of shape [n, k], the result has shape [m*n, k].
    ///
    /// # Definition
    ///
    /// ```text
    /// (A ⊙ B)[:, j] = A[:, j] ⊗ B[:, j]
    /// ```
    ///
    /// Each column of the result is the Kronecker product of the corresponding
    /// columns of A and B.
    ///
    /// # Properties
    ///
    /// - (A ⊙ B)^T (A ⊙ B) = (A^T A) * (B^T B) (Hadamard/element-wise product)
    /// - Essential for CP/PARAFAC tensor decomposition
    /// - Related to mode-n unfolding operations
    ///
    /// # Use Cases
    ///
    /// - CP/PARAFAC tensor decomposition (ALS updates)
    /// - Tucker decomposition
    /// - Multi-linear algebra
    /// - Compressed sensing
    fn khatri_rao(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

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

    /// Convert Real Schur form to Complex Schur form: rsf2csf
    ///
    /// Transforms the real Schur decomposition (with 2×2 blocks for complex
    /// eigenvalue pairs) into the complex Schur form where T is truly upper
    /// triangular with complex eigenvalues on the diagonal.
    ///
    /// # Algorithm
    ///
    /// For each 2×2 block in T corresponding to complex conjugate eigenvalues a ± bi:
    /// 1. Compute eigenvalues of the 2×2 block
    /// 2. Construct a 2×2 unitary rotation that diagonalizes the block
    /// 3. Apply the rotation to T and accumulate into Z
    ///
    /// # Input
    ///
    /// Takes the output of `schur_decompose` (real Schur form).
    ///
    /// # Output
    ///
    /// Returns complex Schur form where:
    /// - T is upper triangular (not quasi-triangular)
    /// - Eigenvalues appear on diagonal of T
    /// - Z is unitary
    fn rsf2csf(
        &self,
        schur: &super::SchurDecomposition<R>,
    ) -> Result<super::ComplexSchurDecomposition<R>>;

    /// Generalized Schur (QZ) decomposition for matrix pencil (A, B)
    ///
    /// Computes the QZ decomposition: A = Q @ S @ Z^H, B = Q @ T @ Z^H
    ///
    /// # Algorithm
    ///
    /// 1. Reduce (A, B) to Hessenberg-triangular form (H, T)
    /// 2. Apply QZ iteration to reduce H to quasi-triangular S
    /// 3. Extract generalized eigenvalues from diagonal blocks
    ///
    /// # Generalized Eigenvalues
    ///
    /// The generalized eigenvalues λ satisfy: det(A - λB) = 0
    /// Computed as: λ_i = alpha_i / beta_i where alpha and beta are
    /// extracted from the diagonal of S and T respectively.
    ///
    /// # Requirements
    ///
    /// Both A and B must be square matrices of the same size.
    fn qz_decompose(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
    ) -> Result<super::GeneralizedSchurDecomposition<R>>;

    /// Polar decomposition: A = U @ P
    ///
    /// Decomposes matrix A into a unitary matrix U and a positive semi-definite
    /// Hermitian matrix P.
    ///
    /// # Algorithm
    ///
    /// Uses the Newton iteration for matrix sign function:
    /// 1. Compute SVD: A = U_svd @ S @ V^H
    /// 2. U = U_svd @ V^H (unitary factor)
    /// 3. P = V @ S @ V^H (positive semi-definite factor)
    ///
    /// Alternative: Newton iteration X_{k+1} = (X_k + X_k^{-H}) / 2
    ///
    /// # Properties
    ///
    /// - For invertible A: U is the closest unitary matrix to A (in Frobenius norm)
    /// - P is unique and equals sqrt(A^H @ A)
    /// - For real matrices: U is orthogonal, P is symmetric positive semi-definite
    fn polar_decompose(&self, a: &Tensor<R>) -> Result<super::PolarDecomposition<R>>;
}

/// Tensor decomposition algorithms for higher-order tensors (N-dimensional arrays)
///
/// This trait provides algorithms for decomposing N-dimensional tensors into
/// structured forms. Unlike matrix decompositions which operate on 2D arrays,
/// tensor decompositions handle arbitrary-dimensional data.
///
/// # Core Operations
///
/// The trait provides fundamental tensor operations that are building blocks
/// for decomposition algorithms:
///
/// - **Mode-n Unfolding (Matricization)**: Convert N-D tensor to 2D matrix
/// - **Mode-n Folding**: Inverse of unfolding, reconstruct tensor from matrix
/// - **Mode-n Product**: Multiply tensor by matrix along a specific mode
///
/// # Decomposition Algorithms
///
/// - **Tucker**: T ≈ G ×₁ A₁ ×₂ A₂ ... (core tensor + factor matrices)
/// - **HOSVD**: Higher-Order SVD (Tucker with orthogonal factors via SVD)
/// - **CP/PARAFAC**: T ≈ Σᵣ λᵣ (a₁ʳ ⊗ a₂ʳ ⊗ ...) (sum of rank-1 tensors)
/// - **Tensor-Train**: T = G₁ × G₂ × ... × Gₙ (sequence of 3D cores)
///
/// # Implementation Requirements
///
/// All backends must implement these algorithms identically to ensure
/// numerical parity. The algorithms use existing operations (SVD, matmul,
/// reshape, permute) from the runtime.
///
/// # Use Cases
///
/// - **Data Compression**: Reduce storage for multi-dimensional arrays
/// - **Dimensionality Reduction**: Extract principal components from tensor data
/// - **Latent Factor Models**: Discover hidden structure in multi-way data
/// - **Quantum Systems**: Tensor network representations
/// - **Recommender Systems**: User-item-context factorization
pub trait TensorDecomposeAlgorithms<R: Runtime>: LinearAlgebraAlgorithms<R> {
    /// Mode-n unfolding (matricization) of a tensor
    ///
    /// Unfolds an N-dimensional tensor into a 2D matrix by arranging mode-n
    /// fibers as columns of the resulting matrix.
    ///
    /// # Mathematical Definition
    ///
    /// For a tensor T of shape [I₁, I₂, ..., Iₙ], the mode-n unfolding T₍ₙ₎
    /// is a matrix of shape [Iₙ, ∏ⱼ≠ₙ Iⱼ] where:
    ///
    /// ```text
    /// T₍ₙ₎[iₙ, j] = T[i₁, i₂, ..., iₙ, ..., iₙ]
    /// ```
    ///
    /// where j is computed from indices (i₁, ..., iₙ₋₁, iₙ₊₁, ..., iₙ)
    /// using a specific ordering convention.
    ///
    /// # Convention
    ///
    /// Uses the standard convention where modes are ordered as:
    /// n, n+1, ..., N, 1, 2, ..., n-1 (forward cyclic from mode n)
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor of arbitrary dimension (≥ 2)
    /// * `mode` - Mode along which to unfold (0-indexed, must be < ndim)
    ///
    /// # Returns
    ///
    /// Matrix of shape [I_mode, ∏ⱼ≠mode Iⱼ]
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Tensor of shape [2, 3, 4]
    /// let unfolded = client.unfold(&tensor, 1)?;
    /// // Result has shape [3, 8] (mode-1 fibers as rows)
    /// ```
    fn unfold(&self, tensor: &Tensor<R>, mode: usize) -> Result<Tensor<R>>;

    /// Mode-n folding (tensorization) - inverse of unfolding
    ///
    /// Reconstructs an N-dimensional tensor from its mode-n unfolding.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The mode-n unfolded matrix [I_mode, ∏ⱼ≠mode Iⱼ]
    /// * `mode` - Mode that was unfolded (0-indexed)
    /// * `shape` - Original tensor shape [I₁, I₂, ..., Iₙ]
    ///
    /// # Returns
    ///
    /// Tensor of the specified shape
    ///
    /// # Panics
    ///
    /// If matrix dimensions don't match the expected unfolded size for the given shape.
    fn fold(&self, matrix: &Tensor<R>, mode: usize, shape: &[usize]) -> Result<Tensor<R>>;

    /// Mode-n product: tensor × matrix along mode n
    ///
    /// Multiplies a tensor by a matrix along a specified mode. This is the
    /// fundamental operation used in Tucker decomposition reconstruction.
    ///
    /// # Mathematical Definition
    ///
    /// For tensor T of shape [I₁, ..., Iₙ, ..., Iₙ] and matrix M of shape [J, Iₙ],
    /// the mode-n product Y = T ×ₙ M has shape [I₁, ..., J, ..., Iₙ] where:
    ///
    /// ```text
    /// Y[i₁, ..., j, ..., iₙ] = Σₖ T[i₁, ..., k, ..., iₙ] × M[j, k]
    /// ```
    ///
    /// # Equivalent Operations
    ///
    /// ```text
    /// T ×ₙ M  ⟺  fold(M @ unfold(T, n), n, new_shape)
    /// ```
    ///
    /// # Properties
    ///
    /// - (T ×ₘ A) ×ₙ B = (T ×ₙ B) ×ₘ A  when m ≠ n (modes commute)
    /// - (T ×ₙ A) ×ₙ B = T ×ₙ (BA)  (same mode contracts)
    /// - T ×ₙ I = T  (identity matrix leaves tensor unchanged)
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor of shape [I₁, ..., Iₙ, ..., Iₙ]
    /// * `matrix` - Matrix of shape [J, Iₙ] to multiply along mode n
    /// * `mode` - Mode along which to multiply (0-indexed)
    ///
    /// # Returns
    ///
    /// Tensor of shape [I₁, ..., J, ..., Iₙ] (mode n dimension changed from Iₙ to J)
    fn mode_n_product(
        &self,
        tensor: &Tensor<R>,
        matrix: &Tensor<R>,
        mode: usize,
    ) -> Result<Tensor<R>>;

    /// Higher-Order SVD (HOSVD) decomposition
    ///
    /// Computes a Tucker decomposition where factor matrices are orthogonal,
    /// obtained by computing truncated SVD of each mode-n unfolding.
    ///
    /// # Algorithm
    ///
    /// 1. For each mode n = 1, ..., N:
    ///    - Compute mode-n unfolding: T₍ₙ₎
    ///    - Compute truncated SVD: T₍ₙ₎ ≈ Uₙ @ Sₙ @ Vₙᵀ
    ///    - Set factor matrix Aₙ = first Rₙ columns of Uₙ
    /// 2. Compute core: G = T ×₁ A₁ᵀ ×₂ A₂ᵀ ... ×ₙ Aₙᵀ
    ///
    /// # Properties
    ///
    /// - Factor matrices are orthogonal: Aₙᵀ @ Aₙ = I
    /// - Core tensor is "all-orthogonal": mode-n unfoldings have orthogonal rows
    /// - NOT the best rank-(R₁, ..., Rₙ) approximation (use Tucker ALS for that)
    /// - Fast: O(N × SVD cost) vs iterative methods
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor of arbitrary dimension
    /// * `ranks` - Multilinear ranks [R₁, R₂, ..., Rₙ]. Each Rₖ ≤ Iₖ.
    ///             Use 0 or dimension size to keep full rank for that mode.
    ///
    /// # Returns
    ///
    /// Tucker decomposition with orthogonal factor matrices
    fn hosvd(&self, tensor: &Tensor<R>, ranks: &[usize]) -> Result<super::TuckerDecomposition<R>>;

    /// Tucker decomposition via Higher-Order Orthogonal Iteration (HOOI)
    ///
    /// Iteratively refines a Tucker decomposition to minimize reconstruction error.
    /// More accurate than HOSVD but more expensive.
    ///
    /// # Algorithm
    ///
    /// 1. Initialize factors using HOSVD or random
    /// 2. Repeat until convergence:
    ///    - For each mode n:
    ///      - Compute: Y = T ×₁ A₁ᵀ ... ×ₙ₋₁ Aₙ₋₁ᵀ ×ₙ₊₁ Aₙ₊₁ᵀ ... ×ₙ Aₙᵀ
    ///      - Update Aₙ = leading Rₙ left singular vectors of unfold(Y, n)
    /// 3. Compute core: G = T ×₁ A₁ᵀ ×₂ A₂ᵀ ... ×ₙ Aₙᵀ
    ///
    /// # Convergence
    ///
    /// - Always converges (monotonically decreasing error)
    /// - May converge to local minimum
    /// - Typically converges in 5-20 iterations
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor
    /// * `ranks` - Multilinear ranks [R₁, R₂, ..., Rₙ]
    /// * `options` - Algorithm options (max_iter, tolerance, initialization)
    ///
    /// # Returns
    ///
    /// Tucker decomposition with approximately orthogonal factors
    fn tucker(
        &self,
        tensor: &Tensor<R>,
        ranks: &[usize],
        options: super::TuckerOptions,
    ) -> Result<super::TuckerDecomposition<R>>;

    /// CP/PARAFAC decomposition via Alternating Least Squares (ALS)
    ///
    /// Decomposes a tensor as a sum of rank-one tensors using the ALS algorithm.
    ///
    /// # Algorithm
    ///
    /// 1. Initialize factor matrices randomly or via SVD
    /// 2. Repeat until convergence:
    ///    - For each mode n:
    ///      - Fix all factors except Aₙ
    ///      - Solve least squares for Aₙ:
    ///        Aₙ = T₍ₙ₎ @ (⊙ⱼ≠ₙ Aⱼ) @ (⊛ⱼ≠ₙ AⱼᵀAⱼ)⁻¹
    ///    - Optionally normalize factors and update weights
    /// 3. Return factor matrices and weights
    ///
    /// # Uniqueness
    ///
    /// CP decomposition is essentially unique (up to permutation and scaling)
    /// under Kruskal's condition:
    ///
    /// ```text
    /// krank(A₁) + krank(A₂) + ... + krank(Aₙ) ≥ 2R + (N - 1)
    /// ```
    ///
    /// where krank is the Kruskal rank.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor
    /// * `rank` - CP rank (number of rank-1 components)
    /// * `options` - Algorithm options (max_iter, tolerance, initialization)
    ///
    /// # Returns
    ///
    /// CP decomposition with factor matrices and weights
    fn cp_decompose(
        &self,
        tensor: &Tensor<R>,
        rank: usize,
        options: super::CpOptions,
    ) -> Result<super::CpDecomposition<R>>;

    /// Tensor-Train (TT) decomposition via TT-SVD
    ///
    /// Decomposes a tensor into a train of 3D core tensors connected by
    /// contractions, using sequential SVD.
    ///
    /// # Algorithm (TT-SVD)
    ///
    /// 1. Reshape T to [I₁, I₂ × ... × Iₙ]
    /// 2. Compute truncated SVD: T ≈ U @ S @ Vᵀ, keep rank R₁
    /// 3. Set G₁ = reshape(U, [1, I₁, R₁])
    /// 4. Reshape S @ Vᵀ to [R₁ × I₂, I₃ × ... × Iₙ]
    /// 5. Repeat SVD for each mode
    /// 6. Last core Gₙ has shape [Rₙ₋₁, Iₙ, 1]
    ///
    /// # Rank Selection
    ///
    /// TT-ranks are determined by the `tolerance` parameter:
    /// - Keep singular values until cumulative truncation error < tolerance × ||T||
    /// - Or use `max_rank` to cap the maximum TT-rank
    ///
    /// # Quasi-Optimal
    ///
    /// TT-SVD is quasi-optimal: the error is at most √(N-1) times
    /// the best possible error for the given ranks.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Input tensor
    /// * `max_rank` - Maximum allowed TT-rank (0 for no limit)
    /// * `tolerance` - Relative tolerance for rank truncation
    ///
    /// # Returns
    ///
    /// Tensor-Train decomposition with sequence of 3D cores
    fn tensor_train(
        &self,
        tensor: &Tensor<R>,
        max_rank: usize,
        tolerance: f64,
    ) -> Result<super::TensorTrainDecomposition<R>>;

    /// Reconstruct tensor from Tucker decomposition
    ///
    /// Computes: T = G ×₁ A₁ ×₂ A₂ ... ×ₙ Aₙ
    ///
    /// # Arguments
    ///
    /// * `decomp` - Tucker decomposition (core + factor matrices)
    ///
    /// # Returns
    ///
    /// Full tensor of shape [I₁, I₂, ..., Iₙ]
    fn tucker_reconstruct(&self, decomp: &super::TuckerDecomposition<R>) -> Result<Tensor<R>>;

    /// Reconstruct tensor from CP decomposition
    ///
    /// Computes: T = Σᵣ λᵣ × (a₁ʳ ⊗ a₂ʳ ⊗ ... ⊗ aₙʳ)
    ///
    /// # Arguments
    ///
    /// * `decomp` - CP decomposition (factor matrices + weights)
    /// * `shape` - Output tensor shape [I₁, I₂, ..., Iₙ]
    ///
    /// # Returns
    ///
    /// Full tensor of the specified shape
    fn cp_reconstruct(
        &self,
        decomp: &super::CpDecomposition<R>,
        shape: &[usize],
    ) -> Result<Tensor<R>>;

    /// Reconstruct tensor from Tensor-Train decomposition
    ///
    /// Contracts all TT-cores to recover the full tensor.
    ///
    /// # Arguments
    ///
    /// * `decomp` - Tensor-Train decomposition (sequence of 3D cores)
    ///
    /// # Returns
    ///
    /// Full tensor of shape [I₁, I₂, ..., Iₙ]
    fn tt_reconstruct(&self, decomp: &super::TensorTrainDecomposition<R>) -> Result<Tensor<R>>;
}
