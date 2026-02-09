//! Decomposition result types for linear algebra operations
//!
//! This module contains the result structures returned by matrix decompositions:
//! LU, Cholesky, QR, SVD, Eigendecomposition, Schur decomposition, and
//! higher-order tensor decompositions (Tucker, CP/PARAFAC, Tensor-Train).

use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// LU decomposition result: PA = LU
///
/// L is lower triangular with unit diagonal, U is upper triangular.
/// P is a permutation matrix (stored as pivot indices).
pub struct LuDecomposition<R: Runtime> {
    /// Lower triangular factor L (unit diagonal, stored in lower part)
    /// Upper triangular factor U (stored in upper part including diagonal)
    /// Stored together in a single matrix for memory efficiency.
    pub lu: Tensor<R>,

    /// Pivot indices: row `i` was swapped with row `pivots[i]`
    /// I64 tensor of shape `[min(m, n)]`
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
    /// Orthogonal matrix Q `[m, m]` or `[m, k]` for thin QR
    pub q: Tensor<R>,

    /// Upper triangular matrix R `[m, n]` or `[k, n]` for thin QR
    pub r: Tensor<R>,
}

/// Singular Value Decomposition result: A = U @ S @ V^T
pub struct SvdDecomposition<R: Runtime> {
    /// Left singular vectors U `[m, k]` where k = min(m, n)
    pub u: Tensor<R>,

    /// Singular values `S` `[k]` (sorted in descending order)
    pub s: Tensor<R>,

    /// Right singular vectors V^T `[k, n]`
    pub vt: Tensor<R>,
}

/// Eigendecomposition result for symmetric matrices: A = V @ diag(λ) @ V^T
///
/// For a real symmetric matrix A, all eigenvalues are real and eigenvectors
/// form an orthonormal basis. The decomposition satisfies:
/// - A @ `V[:,i]` = `λ[i]` * `V[:,i]` for each eigenpair
/// - V^T @ V = I (eigenvectors are orthonormal)
/// - A = V @ diag(λ) @ V^T
pub struct EigenDecomposition<R: Runtime> {
    /// Eigenvalues `λ` `[n]` (sorted in descending order by magnitude)
    pub eigenvalues: Tensor<R>,

    /// Eigenvector matrix `V` `[n, n]` where `V[:,i]` is the eigenvector for `λ[i]`
    /// Columns are orthonormal: `V^T @ V = I`
    pub eigenvectors: Tensor<R>,
}

/// Schur decomposition result: A = Z @ T @ Z^T
///
/// For a real square matrix A, the Schur decomposition factors A into:
/// - `Z`: orthogonal matrix (`Z^T @ Z = I`)
/// - `T`: upper quasi-triangular (real Schur form)
///
/// The real Schur form `T` has:
/// - Real eigenvalues on the diagonal (1×1 blocks)
/// - Complex conjugate eigenvalue pairs in 2×2 blocks on the diagonal
///
/// # Properties
///
/// - `A = Z @ T @ Z^T` (reconstruction)
/// - `Z^T @ Z = I` (orthogonality)
/// - Eigenvalues of A are the diagonal blocks of `T`
/// - For symmetric A, `T` is diagonal (eigenvalue decomposition)
///
/// # Use Cases
///
/// - Matrix function computation (expm, logm, sqrtm)
/// - Solving matrix equations (Sylvester, Lyapunov)
/// - Stability analysis of dynamical systems
pub struct SchurDecomposition<R: Runtime> {
    /// Orthogonal transformation matrix `Z` `[n, n]`
    /// Columns form an orthonormal basis: `Z^T @ Z = I`
    pub z: Tensor<R>,

    /// Upper quasi-triangular Schur form `T` `[n, n]`
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
/// - `eigenvalues_real[i]` = `Re(λ_i)`
/// - `eigenvalues_imag[i]` = `Im(λ_i)`
///
/// For real eigenvalues, `eigenvalues_imag[i] = 0`.
/// For complex conjugate pairs, they appear consecutively:
/// - `λ_j = a + bi` (`eigenvalues_real[j] = a`, `eigenvalues_imag[j] = b`)
/// - `λ_{j+1} = a - bi` (`eigenvalues_real[j+1] = a`, `eigenvalues_imag[j+1] = -b`)
///
/// # Eigenvector Storage
///
/// Eigenvectors are stored as separate real and imaginary matrices.
/// For a real eigenvalue at index `j`, the eigenvector is:
/// - `eigenvectors_real[:, j]` (imaginary part is zero)
///
/// For complex conjugate pairs at indices `j` and `j+1`:
/// - Eigenvector for `λ_j = a + bi` is: `eigenvectors_real[:, j] + i * eigenvectors_imag[:, j]`
/// - Eigenvector for `λ_{j+1} = a - bi` is: `eigenvectors_real[:, j] - i * eigenvectors_imag[:, j]`
pub struct GeneralEigenDecomposition<R: Runtime> {
    /// Real parts of eigenvalues `[n]`
    pub eigenvalues_real: Tensor<R>,

    /// Imaginary parts of eigenvalues `[n]`
    /// Zero for real eigenvalues, non-zero for complex conjugate pairs
    pub eigenvalues_imag: Tensor<R>,

    /// Real parts of eigenvector matrix `[n, n]`
    /// Column `j` is the real part of eigenvector for `λ_j`
    pub eigenvectors_real: Tensor<R>,

    /// Imaginary parts of eigenvector matrix `[n, n]`
    /// Column `j` is the imaginary part of eigenvector for `λ_j`
    /// Zero for real eigenvalues
    pub eigenvectors_imag: Tensor<R>,
}

/// Complex Schur decomposition result: A = Z @ T @ Z^H
///
/// Converts the real Schur form to complex Schur form where T is truly
/// upper triangular (not quasi-triangular) with complex eigenvalues on
/// the diagonal.
///
/// # Properties
///
/// - Z is unitary: Z^H @ Z = I
/// - T is upper triangular with eigenvalues on diagonal
/// - A = Z @ T @ Z^H (reconstruction)
///
/// # Storage Format
///
/// Both Z and T are stored as separate real and imaginary parts to
/// support backends without native complex number support.
pub struct ComplexSchurDecomposition<R: Runtime> {
    /// Real part of unitary matrix Z `[n, n]`
    pub z_real: Tensor<R>,

    /// Imaginary part of unitary matrix Z `[n, n]`
    pub z_imag: Tensor<R>,

    /// Real part of upper triangular matrix T `[n, n]`
    pub t_real: Tensor<R>,

    /// Imaginary part of upper triangular matrix T `[n, n]`
    pub t_imag: Tensor<R>,
}

/// Generalized Schur (QZ) decomposition result: A = Q @ S @ Z^H, B = Q @ T @ Z^H
///
/// For a pair of matrices (A, B), the QZ decomposition factors them as:
/// - A = Q @ S @ Z^H
/// - B = Q @ T @ Z^H
///
/// where Q and Z are unitary, S is upper quasi-triangular, and T is upper triangular.
///
/// # Generalized Eigenvalues
///
/// The generalized eigenvalues λ satisfy det(A - λB) = 0 and are computed as
/// the ratios of diagonal elements: `λ_i = S[i,i] / T[i,i]` (when `T[i,i]` ≠ 0).
///
/// When `T[i,i]` = 0, the eigenvalue is infinite.
///
/// # Use Cases
///
/// - Generalized eigenvalue problems: Av = λBv
/// - Control theory: descriptor systems
/// - Stability analysis of matrix pencils
pub struct GeneralizedSchurDecomposition<R: Runtime> {
    /// Left unitary matrix Q `[n, n]`
    pub q: Tensor<R>,

    /// Right unitary matrix Z `[n, n]`
    pub z: Tensor<R>,

    /// Upper quasi-triangular matrix S (from A) `[n, n]`
    pub s: Tensor<R>,

    /// Upper triangular matrix T (from B) `[n, n]`
    pub t: Tensor<R>,

    /// Real parts of generalized eigenvalues `[n]`
    pub eigenvalues_real: Tensor<R>,

    /// Imaginary parts of generalized eigenvalues `[n]`
    pub eigenvalues_imag: Tensor<R>,
}

/// Polar decomposition result: A = U @ P
///
/// Every square matrix A can be uniquely factored as the product of a
/// unitary matrix U and a positive semi-definite Hermitian matrix P.
///
/// # Properties
///
/// - U^H @ U = I (U is unitary/orthogonal for real matrices)
/// - P = P^H and all eigenvalues of P are non-negative (P is positive semi-definite)
/// - For invertible A: P = sqrt(A^H @ A)
/// - A = U @ P (right polar decomposition)
///
/// # Use Cases
///
/// - Extracting rotation from transformation matrices
/// - Closest orthogonal matrix to A
/// - Strain decomposition in continuum mechanics
/// - Procrustes problems
pub struct PolarDecomposition<R: Runtime> {
    /// Unitary/orthogonal matrix U `[n, n]`
    /// For real A: U is orthogonal (U^T @ U = I)
    /// For complex A: U is unitary (U^H @ U = I)
    pub u: Tensor<R>,

    /// Positive semi-definite Hermitian matrix P `[n, n]`
    /// P = sqrt(A^H @ A) for right polar decomposition
    pub p: Tensor<R>,
}

// ============================================================================
// Tensor Decompositions (Higher-Order Tensors)
// ============================================================================

/// Tucker decomposition result: T ≈ G ×₁ A₁ ×₂ A₂ ×₃ A₃ ...
///
/// The Tucker decomposition factors an N-dimensional tensor T into a smaller
/// core tensor G multiplied by a factor matrix along each mode.
///
/// # Mathematical Definition
///
/// For a tensor T of shape `[I₁, I₂, ..., Iₙ]`, the Tucker decomposition is:
///
/// ```text
/// T ≈ G ×₁ A₁ ×₂ A₂ ... ×ₙ Aₙ
/// ```
///
/// where:
/// - `G` is the core tensor of shape `[R₁, R₂, ..., Rₙ]` (the multilinear ranks)
/// - `Aₖ` are factor matrices of shape `[Iₖ, Rₖ]` for each mode `k`
/// - ×ₖ denotes mode-`k` product (tensor contracted with matrix along mode `k`)
///
/// # Element-wise Formula
///
/// ```text
/// T[i₁, i₂, ..., iₙ] ≈ Σ_{r₁,r₂,...,rₙ} G[r₁, r₂, ..., rₙ] × A₁[i₁, r₁] × A₂[i₂, r₂] × ... × Aₙ[iₙ, rₙ]
/// ```
///
/// # Properties
///
/// - **Compression**: Core tensor is smaller than original (compression ratio: `∏Rₖ / ∏Iₖ`)
/// - **Orthogonality**: Factor matrices can be orthogonal (HOSVD gives orthogonal factors)
/// - **Uniqueness**: NOT unique (rotation of factors and inverse rotation of core)
/// - **Nesting**: Lower-rank Tucker fits inside higher-rank Tucker (unlike CP)
///
/// # Comparison with CP
///
/// | Aspect | Tucker | CP/PARAFAC |
/// |--------|--------|------------|
/// | Core | Dense tensor `G` | Diagonal (`λ` weights) |
/// | Parameters | `∏Rₖ + ΣIₖRₖ` | `R × (1 + ΣIₖ)` |
/// | Uniqueness | Not unique | Unique under mild conditions |
/// | Flexibility | Flexible (asymmetric ranks) | Symmetric rank R |
///
/// # Use Cases
///
/// - Image/video compression
/// - Dimensionality reduction for multi-way data
/// - Feature extraction from tensor data
/// - Subspace learning
pub struct TuckerDecomposition<R: Runtime> {
    /// Core tensor `G` of shape `[R₁, R₂, ..., Rₙ]`
    ///
    /// Contains the compressed representation of the tensor.
    /// The multilinear ranks `Rₖ` control the approximation quality.
    pub core: Tensor<R>,

    /// Factor matrices `[A₁, A₂, ..., Aₙ]`, one per mode
    ///
    /// Each `Aₖ` has shape `[Iₖ, Rₖ]` where `Iₖ` is the original dimension
    /// and `Rₖ` is the chosen rank for mode `k`.
    ///
    /// For HOSVD, these are orthogonal matrices (`Aₖᵀ @ Aₖ = I`).
    pub factors: Vec<Tensor<R>>,
}

/// Options for Tucker decomposition algorithms.
#[derive(Debug, Clone)]
pub struct TuckerOptions {
    /// Maximum number of iterations for iterative algorithms (ALS, HOOI).
    /// Default: 100
    pub max_iter: usize,

    /// Convergence tolerance. Algorithm stops when relative change < tolerance.
    /// Default: 1e-6
    pub tolerance: f64,

    /// Initialization method for factor matrices.
    /// Default: `TuckerInit::Hosvd`
    pub init: TuckerInit,
}

/// Initialization methods for Tucker decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TuckerInit {
    /// Higher-Order SVD initialization (recommended).
    ///
    /// Computes truncated SVD of each mode-n unfolding to get orthogonal factors.
    /// Fast and deterministic, often sufficient without further iteration.
    #[default]
    Hosvd,

    /// Random initialization.
    ///
    /// Initialize factors with random orthogonal matrices.
    /// May require more iterations but can escape local minima.
    Random,
}

impl Default for TuckerOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-6,
            init: TuckerInit::Hosvd,
        }
    }
}

/// CP/PARAFAC decomposition result: T ≈ Σᵣ λᵣ × a₁ʳ ⊗ a₂ʳ ⊗ ... ⊗ aₙʳ
///
/// The Canonical Polyadic (CP) decomposition, also known as PARAFAC (Parallel Factor
/// Analysis), factors an N-dimensional tensor as a sum of R rank-one tensors.
///
/// # Mathematical Definition
///
/// For a tensor T of shape `[I₁, I₂, ..., Iₙ]`, the CP decomposition is:
///
/// ```text
/// T ≈ Σᵣ₌₁ᴿ λᵣ × a₁ʳ ⊗ a₂ʳ ⊗ ... ⊗ aₙʳ
/// ```
///
/// where:
/// - R is the CP rank (number of components)
/// - `λᵣ` are scalar weights (can be absorbed into factors)
/// - `aₖʳ` are column vectors of factor matrix `Aₖ` `[Iₖ, R]`
/// - ⊗ denotes outer product
///
/// # Matrix Notation
///
/// Using factor matrices `A₁, A₂, ..., Aₙ` where `Aₖ` has shape `[Iₖ, R]`:
///
/// ```text
/// T_{(1)} ≈ A₁ @ diag(λ) @ (Aₙ ⊙ ... ⊙ A₂)ᵀ
/// ```
///
/// where ⊙ denotes Khatri-Rao (column-wise Kronecker) product.
///
/// # Element-wise Formula
///
/// ```text
/// T[i₁, i₂, ..., iₙ] ≈ Σᵣ₌₁ᴿ λᵣ × A₁[i₁, r] × A₂[i₂, r] × ... × Aₙ[iₙ, r]
/// ```
///
/// # Properties
///
/// - **Uniqueness**: Unique under mild conditions (unlike Tucker)
/// - **Interpretability**: Each component is a rank-1 tensor (separable)
/// - **Compression**: Parameters = `R × (1 + Σ Iₖ)` vs `∏ Iₖ` for full tensor
/// - **Diagonal core**: Equivalent to Tucker with superdiagonal core
///
/// # Use Cases
///
/// - Chemometrics (fluorescence spectroscopy)
/// - Psychometrics (multi-trait, multi-method)
/// - Signal processing (source separation)
/// - Recommender systems (user-item-context)
/// - Topic modeling
pub struct CpDecomposition<R: Runtime> {
    /// Factor matrices `[A₁, A₂, ..., Aₙ]`, one per mode
    ///
    /// Each `Aₖ` has shape `[Iₖ, R]` where `Iₖ` is the dimension along mode `k`
    /// and `R` is the CP rank.
    ///
    /// Column r of each factor matrix corresponds to component r.
    pub factors: Vec<Tensor<R>>,

    /// Weights (lambda) for each rank-1 component `[R]`
    ///
    /// Scalar weights that can optionally be absorbed into factor matrices.
    /// For normalized factors (columns have unit norm), weights capture
    /// the "strength" of each component.
    pub weights: Tensor<R>,
}

/// Options for CP/PARAFAC decomposition algorithms.
#[derive(Debug, Clone)]
pub struct CpOptions {
    /// Maximum number of iterations for ALS.
    /// Default: 100
    pub max_iter: usize,

    /// Convergence tolerance. Algorithm stops when relative fit change < tolerance.
    /// Default: 1e-6
    pub tolerance: f64,

    /// Initialization method.
    /// Default: `CpInit::Random`
    pub init: CpInit,

    /// Whether to normalize factors after each iteration.
    /// When true, factor columns are normalized and norms are absorbed into weights.
    /// Default: true
    pub normalize: bool,
}

/// Initialization methods for CP decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CpInit {
    /// Random initialization (recommended for general use).
    ///
    /// Initialize each factor matrix with random values from N(0, 1).
    /// Multiple random starts may be needed to find good solutions.
    #[default]
    Random,

    /// SVD-based initialization.
    ///
    /// Use truncated SVD of mode-1 unfolding for `A₁`, then solve
    /// for remaining factors. More deterministic but may not be optimal.
    Svd,

    /// HOSVD-based initialization.
    ///
    /// Compute Tucker decomposition with diagonal core, extract factors.
    /// Good starting point but computationally more expensive.
    Hosvd,
}

impl Default for CpOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-6,
            init: CpInit::Random,
            normalize: true,
        }
    }
}

/// Sign and log-absolute-determinant result
///
/// For a square matrix A, slogdet computes sign(det(A)) and log(|det(A)|)
/// separately. This is more numerically stable than computing det(A) directly
/// for large matrices where the determinant may overflow or underflow.
///
/// # Properties
///
/// - `det(A) = sign * exp(logabsdet)`
/// - `sign` is -1, 0, or +1 (scalar tensor, same dtype as input)
/// - `logabsdet` is log(|det(A)|) (scalar tensor, same dtype as input)
/// - If `det(A) = 0`, then `sign = 0` and `logabsdet = -inf`
pub struct SlogdetResult<R: Runtime> {
    /// Sign of the determinant: -1.0, 0.0, or 1.0 (scalar tensor)
    pub sign: Tensor<R>,

    /// Natural logarithm of the absolute value of the determinant (scalar tensor)
    pub logabsdet: Tensor<R>,
}

/// Tensor-Train (TT) decomposition result: T = G₁ × G₂ × ... × Gₙ
///
/// The Tensor-Train decomposition represents an N-dimensional tensor as a
/// sequence of 3-dimensional "core" tensors connected by contractions.
///
/// # Mathematical Definition
///
/// For a tensor T of shape `[I₁, I₂, ..., Iₙ]`, the TT decomposition is:
///
/// ```text
/// T[i₁, i₂, ..., iₙ] = G₁[:, i₁, :] @ G₂[:, i₂, :] @ ... @ Gₙ[:, iₙ, :]
/// ```
///
/// where:
/// - Gₖ is a 3D core tensor of shape `[Rₖ₋₁, Iₖ, Rₖ]`
/// - R₀ = Rₙ = 1 (boundary conditions)
/// - Rₖ are the TT-ranks (internal bond dimensions)
///
/// # Visual Representation
///
/// ```text
/// G₁ --- G₂ --- G₃ --- ... --- Gₙ
///  |      |      |              |
/// i₁     i₂     i₃            iₙ
/// ```
///
/// Each edge represents contraction, each dangling leg is a physical index.
///
/// # Properties
///
/// - **Storage**: O(N × I × R²) vs O(Iⁿ) for full tensor
/// - **Uniqueness**: Unique up to orthogonal transformations between cores
/// - **Operations**: Many operations (addition, contraction) stay in TT format
/// - **Rank bounds**: TT-ranks bounded by min(∏ᵢ₌₁ᵏ Iᵢ, ∏ᵢ₌ₖ₊₁ⁿ Iᵢ)
///
/// # Comparison with Tucker and CP
///
/// | Aspect | TT | Tucker | CP |
/// |--------|-----|--------|-----|
/// | Parameters | O(N I R²) | O(Rⁿ + N I R) | O(N I R) |
/// | Curse of dimensionality | Avoided | Mild | Avoided |
/// | Operations in format | Yes | Limited | Limited |
///
/// # Use Cases
///
/// - Quantum many-body systems (Matrix Product States)
/// - High-dimensional PDEs
/// - Machine learning with tensor networks
/// - Uncertainty quantification
/// - Solving high-dimensional problems
pub struct TensorTrainDecomposition<R: Runtime> {
    /// Sequence of 3D core tensors `[G₁, G₂, ..., Gₙ]`
    ///
    /// Each Gₖ has shape `[Rₖ₋₁, Iₖ, Rₖ]` where:
    /// - Rₖ₋₁ is the left rank (R₀ = 1)
    /// - Iₖ is the physical dimension at mode k
    /// - Rₖ is the right rank (Rₙ = 1)
    ///
    /// The cores are connected by contracting adjacent rank dimensions.
    pub cores: Vec<Tensor<R>>,

    /// TT-ranks `[R₁, R₂, ..., Rₙ₋₁]`
    ///
    /// Internal bond dimensions connecting cores.
    /// Note: `R₀ = Rₙ = 1` are implicit boundary conditions.
    pub ranks: Vec<usize>,
}
