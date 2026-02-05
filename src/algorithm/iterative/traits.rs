//! Iterative solver trait
//!
//! Defines the contract for iterative linear solvers.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

use super::types::{BiCgStabOptions, BiCgStabResult, GmresOptions, GmresResult};

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
    ///
    /// # Algorithm (Right-Preconditioned GMRES with Restarts)
    ///
    /// ```text
    /// x = x0 (or zeros)
    /// for restart_cycle = 1, 2, ... until converged:
    ///     r = b - A @ x
    ///     β = ||r||
    ///     if β < tol: return x
    ///
    ///     v[0] = r / β
    ///     H = zeros(m+1, m)  # Hessenberg matrix
    ///
    ///     for j = 0 to m-1:
    ///         # Apply preconditioner and matrix
    ///         z = M⁻¹ @ v[j]      # Sparse triangular solves (ILU)
    ///         w = A @ z            # Sparse matvec
    ///
    ///         # Arnoldi: orthogonalize against previous vectors
    ///         for i = 0 to j:
    ///             H[i,j] = dot(w, v[i])
    ///             w = w - H[i,j] * v[i]
    ///         H[j+1,j] = ||w||
    ///
    ///         if H[j+1,j] < eps: break  # Lucky breakdown
    ///         v[j+1] = w / H[j+1,j]
    ///
    ///         # Apply Givens rotations to update QR factorization
    ///         # Track residual norm efficiently
    ///
    ///         if residual < tol: break
    ///
    ///     # Solve least squares via back substitution
    ///     y = solve_upper_triangular(R[:j,:j], g[:j])
    ///     # Update solution: x = x + V @ (M⁻¹ @ y)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Sparse coefficient matrix in CSR format [n, n]
    /// * `b` - Right-hand side vector [n]
    /// * `x0` - Optional initial guess [n] (defaults to zeros)
    /// * `options` - Solver options (tolerance, max iterations, preconditioner)
    ///
    /// # Returns
    ///
    /// `GmresResult` containing the solution, iteration count, and convergence status
    ///
    /// # Errors
    ///
    /// - Matrix is not square
    /// - Dimension mismatch between A and b
    /// - Preconditioner factorization fails
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
    ///
    /// # Algorithm
    ///
    /// ```text
    /// x = x0, r = b - A @ x, r_hat = r
    /// ρ = 1, α = 1, ω = 1
    /// v = p = 0
    ///
    /// for iter = 1, 2, ... until converged:
    ///     ρ_new = dot(r_hat, r)
    ///     β = (ρ_new / ρ) * (α / ω)
    ///     p = r + β * (p - ω * v)
    ///
    ///     # Apply preconditioner
    ///     p_hat = M⁻¹ @ p
    ///     v = A @ p_hat
    ///
    ///     α = ρ_new / dot(r_hat, v)
    ///     s = r - α * v
    ///
    ///     if ||s|| < tol: x += α * p_hat; return
    ///
    ///     s_hat = M⁻¹ @ s
    ///     t = A @ s_hat
    ///
    ///     ω = dot(t, s) / dot(t, t)
    ///     x = x + α * p_hat + ω * s_hat
    ///     r = s - ω * t
    ///     ρ = ρ_new
    ///
    ///     if ||r|| < tol: return
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Sparse coefficient matrix in CSR format [n, n]
    /// * `b` - Right-hand side vector [n]
    /// * `x0` - Optional initial guess [n] (defaults to zeros)
    /// * `options` - Solver options
    ///
    /// # Returns
    ///
    /// `BiCgStabResult` containing the solution and convergence info
    fn bicgstab(
        &self,
        a: &CsrData<R>,
        b: &Tensor<R>,
        x0: Option<&Tensor<R>>,
        options: BiCgStabOptions,
    ) -> Result<BiCgStabResult<R>>;
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
