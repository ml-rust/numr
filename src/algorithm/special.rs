//! Special mathematical functions for scientific computing
//!
//! This module defines traits for special functions required by probability
//! distributions, statistics, and scientific applications. These are critical
//! for solvr::stats to implement distributions like normal, gamma, beta, etc.
//!
//! # Functions Provided
//!
//! ## Error Functions (for normal distribution)
//! - [`erf`] - Error function
//! - [`erfc`] - Complementary error function (1 - erf(x))
//! - [`erfinv`] - Inverse error function
//!
//! ## Gamma Functions (for gamma, chi2, t, F distributions)
//! - [`gamma`] - Gamma function Γ(x)
//! - [`lgamma`] - Log-gamma function ln(Γ(x)) (numerically stable)
//! - [`digamma`] - Digamma function ψ(x) = Γ'(x)/Γ(x)
//!
//! ## Beta Functions (for beta distribution)
//! - [`beta`] - Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
//! - [`betainc`] - Regularized incomplete beta function I_x(a,b)
//!
//! ## Incomplete Gamma (for gamma/chi2 CDF)
//! - [`gammainc`] - Lower regularized incomplete gamma P(a,x)
//! - [`gammaincc`] - Upper regularized incomplete gamma Q(a,x) = 1 - P(a,x)
//!
//! # Algorithm Sources
//!
//! Implementations follow well-established numerical algorithms:
//! - Cody's rational approximation for erf/erfc
//! - Lanczos approximation for gamma/lgamma
//! - Continued fraction expansion for incomplete gamma/beta
//! - Newton-Raphson iteration for inverse functions

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

// ============================================================================
// Special Functions Trait
// ============================================================================

/// Special mathematical functions for scientific computing.
///
/// All backends must implement these functions to enable solvr probability
/// distributions and statistical functions.
///
/// # Implementation Notes
///
/// - Functions operate element-wise on tensors
/// - Input validation (domain checks) should return appropriate errors
/// - Numerical stability is critical - use established algorithms
/// - GPU implementations can use the same algorithms as CPU
pub trait SpecialFunctions<R: Runtime> {
    // ========================================================================
    // Error Functions
    // ========================================================================

    /// Compute the error function element-wise.
    ///
    /// ```text
    /// erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
    /// ```
    ///
    /// # Properties
    /// - Domain: all real numbers
    /// - Range: (-1, 1)
    /// - erf(0) = 0
    /// - erf(∞) = 1, erf(-∞) = -1
    /// - erf(-x) = -erf(x) (odd function)
    ///
    /// # Algorithm
    ///
    /// Uses Cody's rational approximation with different polynomials for:
    /// - |x| ≤ 0.5: Taylor series based
    /// - 0.5 < |x| ≤ 4: Rational approximation
    /// - |x| > 4: Asymptotic expansion
    ///
    /// Accuracy: ~15 digits for F64, ~7 digits for F32.
    fn erf(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the complementary error function element-wise.
    ///
    /// ```text
    /// erfc(x) = 1 - erf(x) = (2/√π) ∫ₓ^∞ e^(-t²) dt
    /// ```
    ///
    /// # Why erfc Instead of 1 - erf?
    ///
    /// For large x, erf(x) ≈ 1 and computing 1 - erf(x) loses precision.
    /// erfc(x) computes the small tail directly, maintaining accuracy.
    ///
    /// # Properties
    /// - Domain: all real numbers
    /// - Range: (0, 2)
    /// - erfc(0) = 1
    /// - erfc(∞) = 0, erfc(-∞) = 2
    fn erfc(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the inverse error function element-wise.
    ///
    /// Returns y such that erf(y) = x.
    ///
    /// # Properties
    /// - Domain: (-1, 1)
    /// - Range: all real numbers
    /// - erfinv(0) = 0
    /// - erfinv(erf(x)) = x
    ///
    /// # Algorithm
    ///
    /// Uses rational approximation for the central region and
    /// asymptotic expansion for tails, refined by Newton-Raphson.
    fn erfinv(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Gamma Functions
    // ========================================================================

    /// Compute the gamma function element-wise.
    ///
    /// ```text
    /// Γ(x) = ∫₀^∞ t^(x-1) e^(-t) dt
    /// ```
    ///
    /// # Properties
    /// - Γ(n) = (n-1)! for positive integers
    /// - Γ(1) = 1, Γ(1/2) = √π
    /// - Γ(x+1) = x·Γ(x)
    /// - Has poles at non-positive integers (returns NaN/Inf)
    ///
    /// # Algorithm
    ///
    /// Uses Lanczos approximation with g=7 coefficients for the
    /// reflection formula to handle negative arguments.
    fn gamma(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the log-gamma function element-wise.
    ///
    /// ```text
    /// lgamma(x) = ln(|Γ(x)|)
    /// ```
    ///
    /// # Why lgamma Instead of log(gamma)?
    ///
    /// Γ(x) grows extremely fast (Γ(171) overflows F64).
    /// lgamma computes the logarithm directly without overflow.
    ///
    /// # Properties
    /// - lgamma(1) = lgamma(2) = 0
    /// - lgamma(n) = ln((n-1)!) for positive integers
    /// - Always returns real values (uses |Γ(x)|)
    fn lgamma(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the digamma (psi) function element-wise.
    ///
    /// ```text
    /// ψ(x) = d/dx ln(Γ(x)) = Γ'(x)/Γ(x)
    /// ```
    ///
    /// # Properties
    /// - ψ(1) = -γ (negative Euler-Mascheroni constant ≈ -0.5772)
    /// - ψ(n+1) = ψ(n) + 1/n
    /// - ψ(x+1) = ψ(x) + 1/x
    ///
    /// # Use Cases
    /// - Maximum likelihood estimation for gamma distribution
    /// - Computing Fisher information
    fn digamma(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Beta Functions
    // ========================================================================

    /// Compute the beta function element-wise.
    ///
    /// ```text
    /// B(a, b) = Γ(a)Γ(b)/Γ(a+b) = ∫₀¹ t^(a-1)(1-t)^(b-1) dt
    /// ```
    ///
    /// # Arguments
    /// - `a`: First shape parameter (positive)
    /// - `b`: Second shape parameter (positive)
    ///
    /// # Properties
    /// - B(a,b) = B(b,a) (symmetric)
    /// - B(1,1) = 1
    /// - B(a,b) = (a-1)!(b-1)!/(a+b-1)! for positive integers
    fn beta(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the regularized incomplete beta function element-wise.
    ///
    /// ```text
    /// I_x(a,b) = B(x;a,b)/B(a,b) = (1/B(a,b)) ∫₀ˣ t^(a-1)(1-t)^(b-1) dt
    /// ```
    ///
    /// # Arguments
    /// - `a`: First shape parameter (positive)
    /// - `b`: Second shape parameter (positive)
    /// - `x`: Upper limit of integration, must be in [0, 1]
    ///
    /// # Properties
    /// - I_0(a,b) = 0, I_1(a,b) = 1
    /// - I_x(a,b) = 1 - I_{1-x}(b,a)
    /// - This is the CDF of the beta distribution
    ///
    /// # Algorithm
    ///
    /// Uses continued fraction expansion (Lentz's method) with
    /// appropriate transformations for numerical stability.
    fn betainc(&self, a: &Tensor<R>, b: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Incomplete Gamma Functions
    // ========================================================================

    /// Compute the lower regularized incomplete gamma function.
    ///
    /// ```text
    /// P(a, x) = γ(a,x)/Γ(a) = (1/Γ(a)) ∫₀ˣ t^(a-1) e^(-t) dt
    /// ```
    ///
    /// # Arguments
    /// - `a`: Shape parameter (positive)
    /// - `x`: Upper limit of integration (non-negative)
    ///
    /// # Properties
    /// - P(a, 0) = 0, P(a, ∞) = 1
    /// - This is the CDF of the gamma distribution
    /// - Related to chi-squared CDF: P(k/2, x/2)
    ///
    /// # Algorithm
    ///
    /// Uses series expansion for x < a+1, continued fraction otherwise.
    fn gammainc(&self, a: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the upper regularized incomplete gamma function.
    ///
    /// ```text
    /// Q(a, x) = Γ(a,x)/Γ(a) = 1 - P(a, x) = (1/Γ(a)) ∫ₓ^∞ t^(a-1) e^(-t) dt
    /// ```
    ///
    /// # Why gammaincc Instead of 1 - gammainc?
    ///
    /// For large x, P(a,x) ≈ 1 and computing 1 - P loses precision.
    /// Q(a,x) computes the tail directly for numerical stability.
    fn gammaincc(&self, a: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;
}

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validate that dtype is suitable for special functions.
///
/// Special functions require floating-point types for accurate computation.
pub fn validate_special_dtype(dtype: crate::dtype::DType) -> Result<()> {
    use crate::dtype::DType;
    use crate::error::Error;

    match dtype {
        DType::F32 | DType::F64 => Ok(()),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "special function (requires F32 or F64)",
        }),
    }
}

// ============================================================================
// Mathematical Constants
// ============================================================================

/// Square root of pi: √π ≈ 1.7724538509055159
pub const SQRT_PI: f64 = 1.7724538509055160272981674833411451827975;

/// 2 / √π ≈ 1.1283791670955126 (used in erf)
pub const TWO_OVER_SQRT_PI: f64 = 1.1283791670955125738961589031215451716881;

/// Euler-Mascheroni constant: γ ≈ 0.5772156649015329
pub const EULER_MASCHERONI: f64 = 0.5772156649015328606065120900824024310422;

/// ln(√(2π)) ≈ 0.9189385332046727 (used in Stirling's approximation)
pub const LN_SQRT_2PI: f64 = 0.9189385332046727417803297364056176398614;

// ============================================================================
// Lanczos Coefficients for Gamma Function
// ============================================================================

/// Lanczos approximation coefficients (g=7, n=9).
///
/// These coefficients provide ~15 digits of precision for the gamma function.
/// Source: Numerical Recipes, 3rd Edition.
pub const LANCZOS_G: f64 = 7.0;

/// Lanczos coefficients for g=7.
pub const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
];
