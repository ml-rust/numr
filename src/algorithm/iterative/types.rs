//! Types for iterative solvers
//!
//! Contains options, results, and diagnostic types for GMRES, BiCGSTAB, etc.

use crate::runtime::Runtime;
use crate::tensor::Tensor;

use super::super::sparse_linalg::{IluFillLevel, IluMetrics};

// ============================================================================
// Convergence Diagnostics
// ============================================================================

/// Reason why an iterative solver terminated
///
/// Provides actionable information about solver behavior for diagnostics
/// and adaptive preconditioning strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceReason {
    /// Converged: ||r|| < atol (absolute tolerance met)
    AbsoluteTolerance,

    /// Converged: ||r||/||b|| < rtol (relative tolerance met)
    RelativeTolerance,

    /// Converged: both absolute and relative tolerances met
    BothTolerances,

    /// Converged: ||b|| ≈ 0, so x = 0 is the solution
    ZeroRhs,

    /// Converged: Krylov subspace spans exact solution (lucky breakdown)
    ///
    /// This occurs when the Arnoldi process finds a vector that is
    /// already in the span of previous vectors, indicating the exact
    /// solution lies in the current Krylov subspace.
    LuckyBreakdown,

    /// Did not converge: maximum iterations reached
    ///
    /// **Hint**: Increase `max_iter`, use preconditioning, or loosen tolerances
    MaxIterationsReached,

    /// Did not converge: residual stopped decreasing significantly
    ///
    /// **Hint**: System may be ill-conditioned. Try ILU preconditioning
    /// or increase fill level (ILU(0) → ILU(1) → ILU(2)).
    Stagnation,

    /// Did not converge: numerical breakdown (near-zero pivot or NaN)
    ///
    /// **Hint**: Matrix may be singular or nearly singular. Check Jacobian
    /// computation, add diagonal shift, or use regularization.
    NumericalBreakdown,
}

impl ConvergenceReason {
    /// Returns true if this reason indicates successful convergence
    #[inline]
    pub fn is_converged(&self) -> bool {
        matches!(
            self,
            ConvergenceReason::AbsoluteTolerance
                | ConvergenceReason::RelativeTolerance
                | ConvergenceReason::BothTolerances
                | ConvergenceReason::ZeroRhs
                | ConvergenceReason::LuckyBreakdown
        )
    }

    /// Returns a human-readable hint for non-converged cases
    pub fn hint(&self) -> &'static str {
        match self {
            ConvergenceReason::AbsoluteTolerance
            | ConvergenceReason::RelativeTolerance
            | ConvergenceReason::BothTolerances
            | ConvergenceReason::ZeroRhs
            | ConvergenceReason::LuckyBreakdown => "Converged successfully",

            ConvergenceReason::MaxIterationsReached => {
                "Increase max_iter, use preconditioning, or loosen tolerances"
            }

            ConvergenceReason::Stagnation => {
                "System may be ill-conditioned. Try ILU preconditioning or increase fill level"
            }

            ConvergenceReason::NumericalBreakdown => {
                "Matrix may be singular. Check Jacobian, add diagonal shift, or use regularization"
            }
        }
    }
}

impl std::fmt::Display for ConvergenceReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvergenceReason::AbsoluteTolerance => write!(f, "absolute tolerance met"),
            ConvergenceReason::RelativeTolerance => write!(f, "relative tolerance met"),
            ConvergenceReason::BothTolerances => write!(f, "both tolerances met"),
            ConvergenceReason::ZeroRhs => write!(f, "zero RHS"),
            ConvergenceReason::LuckyBreakdown => write!(f, "lucky breakdown"),
            ConvergenceReason::MaxIterationsReached => write!(f, "max iterations reached"),
            ConvergenceReason::Stagnation => write!(f, "stagnation detected"),
            ConvergenceReason::NumericalBreakdown => write!(f, "numerical breakdown"),
        }
    }
}

/// Diagnostic information collected during GMRES solve
///
/// Useful for understanding solver behavior, tuning parameters,
/// and implementing adaptive strategies.
#[derive(Debug, Clone)]
pub struct GmresDiagnostics {
    /// Relative tolerance used
    pub rtol: f64,

    /// Absolute tolerance used
    pub atol: f64,

    /// Maximum iterations allowed
    pub max_iter: usize,

    /// Restart parameter (Krylov subspace dimension)
    pub restart: usize,

    /// Initial residual norm ||b - A*x0||
    pub initial_residual_norm: f64,

    /// Norm of right-hand side ||b||
    pub rhs_norm: f64,

    /// History of residual norms (if tracking enabled)
    ///
    /// Empty unless `GmresOptions::track_residual_history` is true.
    pub residual_history: Vec<f64>,
}

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

    /// Whether to track residual history for diagnostics (default: false)
    ///
    /// When enabled, `GmresDiagnostics::residual_history` will contain
    /// the residual norm at each iteration. Useful for debugging
    /// convergence issues but adds memory overhead.
    pub track_residual_history: bool,
}

impl Default for GmresOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            restart: 30,
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::None,
            track_residual_history: false,
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
    ///
    /// Equivalent to `reason.is_converged()`. Kept for backwards compatibility.
    pub converged: bool,

    /// Reason for termination with actionable diagnostics
    ///
    /// Use `reason.hint()` to get suggestions for improving convergence.
    pub reason: ConvergenceReason,

    /// Diagnostic information about the solve
    ///
    /// Contains tolerance settings, initial residual, RHS norm,
    /// and optionally residual history.
    pub diagnostics: GmresDiagnostics,
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
// Adaptive GMRES Types
// ============================================================================

/// Parameters for detecting stagnation during iterative solving
///
/// Stagnation occurs when the residual fails to decrease significantly
/// over multiple iterations, indicating the preconditioner is insufficient.
#[derive(Debug, Clone)]
pub struct StagnationParams {
    /// Minimum required residual reduction factor over window (default: 0.1)
    ///
    /// If `residual[i] / residual[i - window_size] > reduction_factor`,
    /// stagnation is detected.
    pub reduction_factor: f64,

    /// Number of iterations to look back for reduction (default: 10)
    pub window_size: usize,

    /// Minimum iterations before checking for stagnation (default: 5)
    pub min_iterations: usize,
}

impl Default for StagnationParams {
    fn default() -> Self {
        Self {
            reduction_factor: 0.1,
            window_size: 10,
            min_iterations: 5,
        }
    }
}

/// Configuration for adaptive preconditioning in GMRES
///
/// Adaptive GMRES automatically upgrades the preconditioner when
/// convergence stagnates, starting with cheaper options and escalating
/// to stronger preconditioners as needed.
#[derive(Debug, Clone)]
pub struct AdaptivePreconditionerOptions {
    /// Initial ILU fill level (default: ILU(0))
    pub initial_level: IluFillLevel,

    /// Maximum ILU fill level to upgrade to (default: ILU(2))
    pub max_level: IluFillLevel,

    /// Stagnation detection parameters
    pub stagnation: StagnationParams,

    /// Maximum number of preconditioner upgrades allowed (default: 2)
    pub max_upgrades: usize,

    /// Whether to restart GMRES after upgrading preconditioner (default: true)
    ///
    /// When true, the Krylov subspace is discarded and GMRES restarts
    /// with the new preconditioner. When false, GMRES continues from
    /// the current solution estimate.
    pub restart_on_upgrade: bool,
}

impl Default for AdaptivePreconditionerOptions {
    fn default() -> Self {
        Self {
            initial_level: IluFillLevel::Zero,
            max_level: IluFillLevel::Two,
            stagnation: StagnationParams::default(),
            max_upgrades: 2,
            restart_on_upgrade: true,
        }
    }
}

// ============================================================================
// CG Options and Result
// ============================================================================

/// Configuration options for Conjugate Gradient solver
#[derive(Debug, Clone)]
pub struct CgOptions {
    /// Maximum number of iterations (default: 1000)
    pub max_iter: usize,
    /// Relative tolerance (default: 1e-10)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-14)
    pub atol: f64,
    /// Preconditioner type (default: None)
    pub preconditioner: PreconditionerType,
}

impl Default for CgOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::None,
        }
    }
}

/// Result of Conjugate Gradient solver
#[derive(Debug, Clone)]
pub struct CgResult<R: Runtime> {
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
// MINRES Options and Result
// ============================================================================

/// Configuration options for MINRES solver
#[derive(Debug, Clone)]
pub struct MinresOptions {
    /// Maximum number of iterations (default: 1000)
    pub max_iter: usize,
    /// Relative tolerance (default: 1e-10)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-14)
    pub atol: f64,
    /// Preconditioner type (default: None)
    pub preconditioner: PreconditionerType,
}

impl Default for MinresOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::None,
        }
    }
}

/// Result of MINRES solver
#[derive(Debug, Clone)]
pub struct MinresResult<R: Runtime> {
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
// CGS Options and Result
// ============================================================================

/// Configuration options for CGS (Conjugate Gradient Squared) solver
#[derive(Debug, Clone)]
pub struct CgsOptions {
    /// Maximum number of iterations (default: 1000)
    pub max_iter: usize,
    /// Relative tolerance (default: 1e-10)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-14)
    pub atol: f64,
    /// Preconditioner type (default: None)
    pub preconditioner: PreconditionerType,
}

impl Default for CgsOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::None,
        }
    }
}

/// Result of CGS solver
#[derive(Debug, Clone)]
pub struct CgsResult<R: Runtime> {
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
// Sparse Eigensolver Types
// ============================================================================

/// Which eigenvalues to compute
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WhichEigenvalues {
    /// Largest eigenvalues by magnitude
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

impl Default for WhichEigenvalues {
    fn default() -> Self {
        WhichEigenvalues::LargestMagnitude
    }
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
    ///
    /// Must be > k (number of requested eigenvalues).
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
    /// Eigenvectors as columns of [n, k] matrix
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
    /// Real parts of eigenvalues [k]
    pub eigenvalues_real: Tensor<R>,
    /// Imaginary parts of eigenvalues [k]
    pub eigenvalues_imag: Tensor<R>,
    /// Eigenvectors as columns of [n, k] matrix
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

// ============================================================================
// Adaptive GMRES Types
// ============================================================================

/// Result of adaptive GMRES solver
#[derive(Debug, Clone)]
pub struct AdaptiveGmresResult<R: Runtime> {
    /// Solution vector x such that Ax ≈ b
    pub solution: Tensor<R>,

    /// Total iterations across all preconditioner levels
    pub total_iterations: usize,

    /// Final residual norm ||b - Ax||
    pub residual_norm: f64,

    /// Whether the solver converged within tolerance
    pub converged: bool,

    /// Final ILU fill level used
    pub final_level: IluFillLevel,

    /// Number of preconditioner upgrades performed
    pub upgrades: usize,

    /// Metrics from each ILU factorization (one per upgrade + initial)
    pub ilu_metrics: Vec<IluMetrics>,

    /// Convergence reason for final termination
    pub reason: ConvergenceReason,
}
