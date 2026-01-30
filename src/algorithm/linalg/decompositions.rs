//! Decomposition result types for linear algebra operations
//!
//! This module contains the result structures returned by matrix decompositions:
//! LU, Cholesky, QR, SVD, Eigendecomposition, and Schur decomposition.

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
    /// Real part of unitary matrix Z [n, n]
    pub z_real: Tensor<R>,

    /// Imaginary part of unitary matrix Z [n, n]
    pub z_imag: Tensor<R>,

    /// Real part of upper triangular matrix T [n, n]
    pub t_real: Tensor<R>,

    /// Imaginary part of upper triangular matrix T [n, n]
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
/// the ratios of diagonal elements: λ_i = S[i,i] / T[i,i] (when T[i,i] ≠ 0).
///
/// When T[i,i] = 0, the eigenvalue is infinite.
///
/// # Use Cases
///
/// - Generalized eigenvalue problems: Av = λBv
/// - Control theory: descriptor systems
/// - Stability analysis of matrix pencils
pub struct GeneralizedSchurDecomposition<R: Runtime> {
    /// Left unitary matrix Q [n, n]
    pub q: Tensor<R>,

    /// Right unitary matrix Z [n, n]
    pub z: Tensor<R>,

    /// Upper quasi-triangular matrix S (from A) [n, n]
    pub s: Tensor<R>,

    /// Upper triangular matrix T (from B) [n, n]
    pub t: Tensor<R>,

    /// Real parts of generalized eigenvalues [n]
    pub eigenvalues_real: Tensor<R>,

    /// Imaginary parts of generalized eigenvalues [n]
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
    /// Unitary/orthogonal matrix U [n, n]
    /// For real A: U is orthogonal (U^T @ U = I)
    /// For complex A: U is unitary (U^H @ U = I)
    pub u: Tensor<R>,

    /// Positive semi-definite Hermitian matrix P [n, n]
    /// P = sqrt(A^H @ A) for right polar decomposition
    pub p: Tensor<R>,
}
