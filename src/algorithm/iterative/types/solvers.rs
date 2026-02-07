//! Options and result types for iterative linear solvers

use crate::algorithm::sparse_linalg::{IluFillLevel, IluMetrics};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

use super::common::{ConvergenceReason, GmresDiagnostics, PreconditionerType};

// ============================================================================
// GMRES
// ============================================================================

/// Configuration options for GMRES solver
#[derive(Debug, Clone)]
pub struct GmresOptions {
    /// Maximum number of iterations (default: 1000)
    pub max_iter: usize,
    /// Restart parameter m - Krylov subspace dimension before restart (default: 30)
    pub restart: usize,
    /// Relative tolerance (default: 1e-10)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-14)
    pub atol: f64,
    /// Preconditioner type (default: None)
    pub preconditioner: PreconditionerType,
    /// Whether to track residual history for diagnostics (default: false)
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
    pub converged: bool,
    /// Reason for termination with actionable diagnostics
    pub reason: ConvergenceReason,
    /// Diagnostic information about the solve
    pub diagnostics: GmresDiagnostics,
}

// ============================================================================
// BiCGSTAB
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
// CG
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
// MINRES
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
// CGS
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
// Adaptive GMRES Result
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

// ============================================================================
// LGMRES
// ============================================================================

/// Configuration options for LGMRES (Loose GMRES) solver
///
/// LGMRES augments the Krylov subspace with error approximation vectors
/// from previous restart cycles, accelerating convergence across restarts.
#[derive(Debug, Clone)]
pub struct LgmresOptions {
    /// Maximum number of iterations (default: 1000)
    pub max_iter: usize,
    /// Inner restart dimension m (default: 30)
    pub restart: usize,
    /// Number of augmentation vectors from previous restarts (default: 2)
    pub k_aug: usize,
    /// Relative tolerance (default: 1e-10)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-14)
    pub atol: f64,
    /// Preconditioner type (default: None)
    pub preconditioner: PreconditionerType,
}

impl Default for LgmresOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            restart: 30,
            k_aug: 2,
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::None,
        }
    }
}

/// Result of LGMRES solver
#[derive(Debug, Clone)]
pub struct LgmresResult<R: Runtime> {
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
// QMR
// ============================================================================

/// Configuration options for QMR (Quasi-Minimal Residual) solver
///
/// QMR uses coupled two-term Lanczos with QR factorization for
/// non-symmetric systems. More stable than BiCGSTAB for some problems.
#[derive(Debug, Clone)]
pub struct QmrOptions {
    /// Maximum number of iterations (default: 1000)
    pub max_iter: usize,
    /// Relative tolerance (default: 1e-10)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-14)
    pub atol: f64,
    /// Preconditioner type (default: None)
    pub preconditioner: PreconditionerType,
}

impl Default for QmrOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            rtol: 1e-10,
            atol: 1e-14,
            preconditioner: PreconditionerType::None,
        }
    }
}

/// Result of QMR solver
#[derive(Debug, Clone)]
pub struct QmrResult<R: Runtime> {
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
// Jacobi
// ============================================================================

/// Configuration options for weighted Jacobi iteration
///
/// Iterates x_{k+1} = x_k + omega * D^{-1} * (b - A*x_k) where D = diag(A).
/// Best for diagonally dominant systems or as a smoother for multigrid.
#[derive(Debug, Clone)]
pub struct JacobiOptions {
    /// Maximum number of iterations (default: 1000)
    pub max_iter: usize,
    /// Relaxation weight (default: 2/3)
    ///
    /// omega = 1.0 is standard Jacobi. omega < 1 is under-relaxation.
    /// omega = 2/3 is optimal for many problems as a smoother.
    pub omega: f64,
    /// Relative tolerance (default: 1e-10)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-14)
    pub atol: f64,
}

impl Default for JacobiOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            omega: 2.0 / 3.0,
            rtol: 1e-10,
            atol: 1e-14,
        }
    }
}

/// Result of Jacobi iteration
#[derive(Debug, Clone)]
pub struct JacobiResult<R: Runtime> {
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
// SOR
// ============================================================================

/// Configuration options for SOR (Successive Over-Relaxation) solver
///
/// SOR performs a forward sweep with relaxation parameter omega.
/// omega = 1 is Gauss-Seidel. 1 < omega < 2 is over-relaxation.
#[derive(Debug, Clone)]
pub struct SorOptions {
    /// Maximum number of iterations (default: 1000)
    pub max_iter: usize,
    /// Relaxation parameter (default: 1.5)
    ///
    /// omega = 1.0 is Gauss-Seidel. 1 < omega < 2 for over-relaxation.
    /// Must be in (0, 2) for convergence on SPD systems.
    pub omega: f64,
    /// Relative tolerance (default: 1e-10)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-14)
    pub atol: f64,
}

impl Default for SorOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            omega: 1.5,
            rtol: 1e-10,
            atol: 1e-14,
        }
    }
}

/// Result of SOR solver
#[derive(Debug, Clone)]
pub struct SorResult<R: Runtime> {
    /// Solution vector x such that Ax ≈ b
    pub solution: Tensor<R>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm ||b - Ax||
    pub residual_norm: f64,
    /// Whether the solver converged within tolerance
    pub converged: bool,
}
