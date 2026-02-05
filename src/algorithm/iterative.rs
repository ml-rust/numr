//! Iterative Solver Contracts
//!
//! This module defines traits for iterative linear solvers used to solve
//! sparse linear systems Ax = b efficiently. These solvers are essential
//! for large-scale problems (100k+ variables) where direct methods would
//! run out of memory.
//!
//! # Solvers
//!
//! - **GMRES**: Generalized Minimal Residual for non-symmetric systems
//! - **BiCGSTAB**: Bi-Conjugate Gradient Stabilized (alternative to GMRES)
//!
//! # Use Cases
//!
//! - Stiff ODE solvers (Radau, BDF) - sparse Jacobian systems
//! - PDE discretizations - finite element/difference systems
//! - Large-scale optimization - Newton systems
//!
//! # Preconditioning
//!
//! Iterative solvers benefit greatly from preconditioning. A preconditioner M
//! approximates A such that M⁻¹A has better spectral properties (eigenvalues
//! clustered near 1). This module supports:
//!
//! - **ILU(0)**: Incomplete LU factorization (via SparseLinAlgAlgorithms)
//! - **IC(0)**: Incomplete Cholesky for SPD systems
//!
//! # Algorithm Guarantees
//!
//! All backends produce identical results (within floating-point tolerance):
//! - Same Krylov subspace construction
//! - Same orthogonalization (Givens rotations)
//! - Same convergence criteria

use crate::error::Result;
use crate::runtime::Runtime;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

// ============================================================================
// Preconditioner Types
// ============================================================================

/// Preconditioner type for iterative solvers
///
/// Preconditioning transforms the system Ax = b to improve convergence:
/// - Right preconditioning: A M⁻¹ (Mx) = b, solve for y = Mx, then x = M⁻¹y
/// - Left preconditioning: M⁻¹A x = M⁻¹b (changes residual definition)
///
/// GMRES uses right preconditioning to preserve the true residual.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PreconditionerType {
    /// No preconditioning (identity preconditioner)
    #[default]
    None,

    /// ILU(0) - Incomplete LU factorization with zero fill-in
    ///
    /// Best for general non-symmetric systems. Computes L and U factors
    /// with the same sparsity pattern as A, then applies M⁻¹x = U⁻¹(L⁻¹x).
    Ilu0,

    /// IC(0) - Incomplete Cholesky factorization with zero fill-in
    ///
    /// Best for symmetric positive definite systems. Computes L such that
    /// A ≈ LLᵀ, then applies M⁻¹x = L⁻ᵀ(L⁻¹x).
    Ic0,
}

// ============================================================================
// GMRES Options and Result
// ============================================================================

/// Configuration options for GMRES solver
#[derive(Debug, Clone)]
pub struct GmresOptions {
    /// Maximum number of iterations (default: 1000)
    pub max_iter: usize,

    /// Restart parameter m - Krylov subspace dimension before restart (default: 30)
    ///
    /// Larger values use more memory (O(n*m)) but may converge faster.
    /// For very ill-conditioned systems, larger m helps.
    /// Typical values: 20-100.
    pub restart: usize,

    /// Relative tolerance (default: 1e-10)
    ///
    /// Convergence when ||b - Ax|| / ||b|| < rtol
    pub rtol: f64,

    /// Absolute tolerance (default: 1e-14)
    ///
    /// Convergence when ||b - Ax|| < atol
    pub atol: f64,

    /// Preconditioner type (default: None)
    pub preconditioner: PreconditionerType,
}

impl Default for GmresOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            restart: 30,
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::None,
        }
    }
}

/// Result of GMRES solver
#[derive(Debug, Clone)]
pub struct GmresResult<R: Runtime> {
    /// Solution vector x such that Ax ≈ b
    pub solution: Tensor<R>,

    /// Number of iterations performed
    pub iterations: usize,

    /// Final residual norm ||b - Ax||
    pub residual_norm: f64,

    /// Whether the solver converged within tolerance
    pub converged: bool,
}

// ============================================================================
// BiCGSTAB Options and Result
// ============================================================================

/// Configuration options for BiCGSTAB solver
#[derive(Debug, Clone)]
pub struct BiCgStabOptions {
    /// Maximum number of iterations (default: 1000)
    pub max_iter: usize,

    /// Relative tolerance (default: 1e-10)
    pub rtol: f64,

    /// Absolute tolerance (default: 1e-14)
    pub atol: f64,

    /// Preconditioner type (default: None)
    pub preconditioner: PreconditionerType,
}

impl Default for BiCgStabOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::None,
        }
    }
}

/// Result of BiCGSTAB solver
#[derive(Debug, Clone)]
pub struct BiCgStabResult<R: Runtime> {
    /// Solution vector x such that Ax ≈ b
    pub solution: Tensor<R>,

    /// Number of iterations performed
    pub iterations: usize,

    /// Final residual norm ||b - Ax||
    pub residual_norm: f64,

    /// Whether the solver converged within tolerance
    pub converged: bool,
}

// ============================================================================
// Iterative Solvers Trait
// ============================================================================

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
    crate::algorithm::SparseLinAlgAlgorithms<R> + crate::ops::LinalgOps<R>
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
