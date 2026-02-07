//! Types for sparse SVD (Lanczos bidiagonalization)

use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Which singular values to compute
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum WhichSingularValues {
    /// Largest singular values
    #[default]
    Largest,
    /// Smallest singular values
    Smallest,
}

/// Configuration options for sparse SVD (Lanczos bidiagonalization)
#[derive(Debug, Clone)]
pub struct SvdsOptions {
    /// Maximum number of iterations (default: 1000)
    pub max_iter: usize,
    /// Convergence tolerance (default: 1e-10)
    pub tol: f64,
    /// Which singular values to compute (default: Largest)
    pub which: WhichSingularValues,
    /// Krylov subspace dimension (default: min(2*k + 1, min(m,n)))
    pub ncv: Option<usize>,
}

impl Default for SvdsOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-10,
            which: WhichSingularValues::default(),
            ncv: None,
        }
    }
}

/// Result of sparse SVD computation
#[derive(Debug, Clone)]
pub struct SparseSvdResult<R: Runtime> {
    /// Left singular vectors U [m, k]
    pub u: Tensor<R>,
    /// Singular values [k] (sorted descending)
    pub singular_values: Tensor<R>,
    /// Right singular vectors Vt [k, n]
    pub vt: Tensor<R>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether all requested singular values converged
    pub converged: bool,
    /// Number of converged singular values
    pub nconv: usize,
}
