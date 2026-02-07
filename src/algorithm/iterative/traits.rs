//! Iterative solver trait
//!
//! Defines the contract for iterative linear solvers.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

use super::types::{
    BiCgStabOptions, BiCgStabResult, CgOptions, CgResult, CgsOptions, CgsResult, GmresOptions,
    GmresResult, JacobiOptions, JacobiResult, LgmresOptions, LgmresResult, MinresOptions,
    MinresResult, QmrOptions, QmrResult, SorOptions, SorResult, SparseEigComplexResult,
    SparseEigOptions, SparseEigResult, SparseSvdResult, SvdsOptions,
};

/// Iterative solvers for sparse linear systems
///
/// This trait defines Krylov subspace methods for solving Ax = b where A is
/// a sparse matrix. All methods support optional preconditioning to improve
/// convergence.
///
/// # Implementation Requirements
///
/// Backends may differ in:
/// - Parallelization strategy
/// - Memory access patterns
///
/// Backends MUST match in:
/// - Mathematical algorithm (same Krylov basis construction)
/// - Orthogonalization method (Givens rotations for numerical stability)
/// - Convergence criteria
pub trait IterativeSolvers<R: Runtime>:
    crate::algorithm::sparse_linalg::SparseLinAlgAlgorithms<R> + crate::ops::LinalgOps<R>
{
    /// GMRES (Generalized Minimal Residual) solver
    ///
    /// Solves Ax = b for non-symmetric sparse systems using Krylov subspace
    /// iteration with Arnoldi orthogonalization.
    fn gmres(
        &self,
        a: &CsrData<R>,
        b: &Tensor<R>,
        x0: Option<&Tensor<R>>,
        options: GmresOptions,
    ) -> Result<GmresResult<R>>;

    /// BiCGSTAB (Bi-Conjugate Gradient Stabilized) solver
    ///
    /// Alternative to GMRES for non-symmetric systems. Uses less memory than
    /// GMRES(m) but may have less predictable convergence.
    fn bicgstab(
        &self,
        a: &CsrData<R>,
        b: &Tensor<R>,
        x0: Option<&Tensor<R>>,
        options: BiCgStabOptions,
    ) -> Result<BiCgStabResult<R>>;

    /// CG (Conjugate Gradient) solver
    ///
    /// Solves Ax = b where A is symmetric positive definite (SPD).
    fn cg(
        &self,
        a: &CsrData<R>,
        b: &Tensor<R>,
        x0: Option<&Tensor<R>>,
        options: CgOptions,
    ) -> Result<CgResult<R>>;

    /// MINRES (Minimum Residual) solver
    ///
    /// Solves Ax = b where A is symmetric (possibly indefinite).
    fn minres(
        &self,
        a: &CsrData<R>,
        b: &Tensor<R>,
        x0: Option<&Tensor<R>>,
        options: MinresOptions,
    ) -> Result<MinresResult<R>>;

    /// CGS (Conjugate Gradient Squared) solver
    ///
    /// Solves Ax = b for non-symmetric systems.
    fn cgs(
        &self,
        a: &CsrData<R>,
        b: &Tensor<R>,
        x0: Option<&Tensor<R>>,
        options: CgsOptions,
    ) -> Result<CgsResult<R>>;

    /// LGMRES (Loose GMRES) solver
    ///
    /// Augments GMRES restarts with error approximation vectors from
    /// previous cycles, improving convergence across restarts.
    fn lgmres(
        &self,
        a: &CsrData<R>,
        b: &Tensor<R>,
        x0: Option<&Tensor<R>>,
        options: LgmresOptions,
    ) -> Result<LgmresResult<R>>;

    /// QMR (Quasi-Minimal Residual) solver
    ///
    /// Uses coupled two-term Lanczos biorthogonalization for non-symmetric systems.
    /// More stable than BiCGSTAB for some problems.
    fn qmr(
        &self,
        a: &CsrData<R>,
        b: &Tensor<R>,
        x0: Option<&Tensor<R>>,
        options: QmrOptions,
    ) -> Result<QmrResult<R>>;

    /// Jacobi (Weighted Jacobi Iteration) solver
    ///
    /// Simple stationary iteration: x_{k+1} = x_k + omega * D^{-1} * (b - Ax_k).
    /// Best for diagonally dominant systems or as a smoother.
    fn jacobi(
        &self,
        a: &CsrData<R>,
        b: &Tensor<R>,
        x0: Option<&Tensor<R>>,
        options: JacobiOptions,
    ) -> Result<JacobiResult<R>>;

    /// SOR (Successive Over-Relaxation) solver
    ///
    /// Forward-sweep SOR with relaxation parameter omega.
    /// omega = 1 is Gauss-Seidel. 1 < omega < 2 for over-relaxation.
    fn sor(
        &self,
        a: &CsrData<R>,
        b: &Tensor<R>,
        x0: Option<&Tensor<R>>,
        options: SorOptions,
    ) -> Result<SorResult<R>>;

    /// Sparse eigensolver for symmetric matrices (Lanczos)
    fn sparse_eig_symmetric(
        &self,
        a: &CsrData<R>,
        k: usize,
        options: SparseEigOptions,
    ) -> Result<SparseEigResult<R>>;

    /// Sparse eigensolver for general (non-symmetric) matrices (Arnoldi/IRAM)
    fn sparse_eig(
        &self,
        a: &CsrData<R>,
        k: usize,
        options: SparseEigOptions,
    ) -> Result<SparseEigComplexResult<R>>;

    /// Sparse SVD via Lanczos bidiagonalization
    ///
    /// Computes the k largest (or smallest) singular values and vectors
    /// of a large sparse matrix.
    fn svds(&self, a: &CsrData<R>, k: usize, options: SvdsOptions) -> Result<SparseSvdResult<R>>;
}

// ============================================================================
// Validation helpers
// ============================================================================

/// Validate inputs for iterative solvers
pub fn validate_iterative_inputs<R: Runtime>(
    a_shape: [usize; 2],
    b: &Tensor<R>,
    x0: Option<&Tensor<R>>,
) -> Result<usize> {
    use crate::error::Error;

    // Check A is square
    let [nrows, ncols] = a_shape;
    if nrows != ncols {
        return Err(Error::ShapeMismatch {
            expected: vec![nrows, nrows],
            got: vec![nrows, ncols],
        });
    }
    let n = nrows;

    // Check b dimension
    if b.ndim() != 1 {
        return Err(Error::Internal(format!(
            "Expected 1D right-hand side, got {}D",
            b.ndim()
        )));
    }
    if b.numel() != n {
        return Err(Error::ShapeMismatch {
            expected: vec![n],
            got: vec![b.numel()],
        });
    }

    // Check x0 if provided
    if let Some(x0) = x0 {
        if x0.ndim() != 1 {
            return Err(Error::Internal(format!(
                "Expected 1D initial guess, got {}D",
                x0.ndim()
            )));
        }
        if x0.numel() != n {
            return Err(Error::ShapeMismatch {
                expected: vec![n],
                got: vec![x0.numel()],
            });
        }
    }

    Ok(n)
}
