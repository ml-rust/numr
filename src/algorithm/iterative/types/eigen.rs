//! Types for sparse eigensolvers (Lanczos and Arnoldi)

use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Which eigenvalues to compute
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum WhichEigenvalues {
    /// Largest eigenvalues by magnitude
    #[default]
    LargestMagnitude,
    /// Smallest eigenvalues by magnitude
    SmallestMagnitude,
    /// Largest eigenvalues by real part
    LargestReal,
    /// Smallest eigenvalues by real part
    SmallestReal,
    /// Eigenvalues closest to the given shift (sigma)
    ClosestTo(f64),
}

/// Configuration options for sparse eigensolvers (Lanczos and Arnoldi)
#[derive(Debug, Clone)]
pub struct SparseEigOptions {
    /// Maximum number of iterations (default: 1000)
    pub max_iter: usize,
    /// Convergence tolerance for eigenvalues (default: 1e-10)
    pub tol: f64,
    /// Which eigenvalues to compute (default: LargestMagnitude)
    pub which: WhichEigenvalues,
    /// Krylov subspace dimension (default: min(2*k + 1, n))
    pub ncv: Option<usize>,
}

impl Default for SparseEigOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-10,
            which: WhichEigenvalues::default(),
            ncv: None,
        }
    }
}

/// Result of sparse eigensolver for symmetric matrices
#[derive(Debug, Clone)]
pub struct SparseEigResult<R: Runtime> {
    /// Eigenvalues (real), sorted according to `which`
    pub eigenvalues: Tensor<R>,
    /// Eigenvectors as columns of `[n, k]` matrix
    pub eigenvectors: Tensor<R>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether all requested eigenvalues converged
    pub converged: bool,
    /// Number of converged eigenvalues
    pub nconv: usize,
}

/// Result of sparse eigensolver for non-symmetric matrices
#[derive(Debug, Clone)]
pub struct SparseEigComplexResult<R: Runtime> {
    /// Real parts of eigenvalues `[k]`
    pub eigenvalues_real: Tensor<R>,
    /// Imaginary parts of eigenvalues `[k]`
    pub eigenvalues_imag: Tensor<R>,
    /// Eigenvectors as columns of `[n, k]` matrix
    ///
    /// For complex conjugate pairs, consecutive columns store the
    /// real and imaginary parts respectively.
    pub eigenvectors: Tensor<R>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether all requested eigenvalues converged
    pub converged: bool,
    /// Number of converged eigenvalues
    pub nconv: usize,
}
