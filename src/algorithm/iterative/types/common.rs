//! Common types: convergence diagnostics, preconditioner selection, stagnation detection

use crate::algorithm::sparse_linalg::IluFillLevel;

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
    pub residual_history: Vec<f64>,
}

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
    Ilu0,

    /// IC(0) - Incomplete Cholesky factorization with zero fill-in
    Ic0,

    /// AMG - Algebraic Multigrid preconditioner
    Amg,
}

/// Parameters for detecting stagnation during iterative solving
#[derive(Debug, Clone)]
pub struct StagnationParams {
    /// Minimum required residual reduction factor over window (default: 0.1)
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
