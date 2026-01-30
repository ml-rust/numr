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
//! ## Bessel Functions
//! - [`bessel_j0`], [`bessel_j1`] - First kind J₀, J₁
//! - [`bessel_y0`], [`bessel_y1`] - Second kind Y₀, Y₁
//! - [`bessel_i0`], [`bessel_i1`] - Modified first kind I₀, I₁
//! - [`bessel_k0`], [`bessel_k1`] - Modified second kind K₀, K₁
//!
//! # Algorithm Sources
//!
//! Implementations follow well-established numerical algorithms:
//! - Cody's rational approximation for erf/erfc
//! - Lanczos approximation for gamma/lgamma
//! - Continued fraction expansion for incomplete gamma/beta
//! - Newton-Raphson iteration for inverse functions
//! - Numerical Recipes polynomial approximations for Bessel functions

pub mod bessel_coefficients;

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
    fn erf(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the complementary error function element-wise.
    ///
    /// ```text
    /// erfc(x) = 1 - erf(x) = (2/√π) ∫ₓ^∞ e^(-t²) dt
    /// ```
    ///
    /// For large x, erf(x) ≈ 1 and computing 1 - erf(x) loses precision.
    /// erfc(x) computes the small tail directly, maintaining accuracy.
    fn erfc(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the inverse error function element-wise.
    ///
    /// Returns y such that erf(y) = x.
    ///
    /// # Properties
    /// - Domain: (-1, 1)
    /// - Range: all real numbers
    /// - erfinv(0) = 0
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
    /// - Has poles at non-positive integers (returns NaN/Inf)
    fn gamma(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the log-gamma function element-wise.
    ///
    /// ```text
    /// lgamma(x) = ln(|Γ(x)|)
    /// ```
    ///
    /// Γ(x) grows extremely fast (Γ(171) overflows F64).
    /// lgamma computes the logarithm directly without overflow.
    fn lgamma(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the digamma (psi) function element-wise.
    ///
    /// ```text
    /// ψ(x) = d/dx ln(Γ(x)) = Γ'(x)/Γ(x)
    /// ```
    fn digamma(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Beta Functions
    // ========================================================================

    /// Compute the beta function element-wise.
    ///
    /// ```text
    /// B(a, b) = Γ(a)Γ(b)/Γ(a+b)
    /// ```
    fn beta(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the regularized incomplete beta function element-wise.
    ///
    /// ```text
    /// I_x(a,b) = B(x;a,b)/B(a,b) = (1/B(a,b)) ∫₀ˣ t^(a-1)(1-t)^(b-1) dt
    /// ```
    fn betainc(&self, a: &Tensor<R>, b: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Incomplete Gamma Functions
    // ========================================================================

    /// Compute the lower regularized incomplete gamma function.
    ///
    /// ```text
    /// P(a, x) = γ(a,x)/Γ(a) = (1/Γ(a)) ∫₀ˣ t^(a-1) e^(-t) dt
    /// ```
    fn gammainc(&self, a: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute the upper regularized incomplete gamma function.
    ///
    /// ```text
    /// Q(a, x) = 1 - P(a, x)
    /// ```
    fn gammaincc(&self, a: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;

    // ========================================================================
    // Bessel Functions
    // ========================================================================

    /// Compute Bessel function of the first kind, order 0.
    ///
    /// J₀(0) = 1, even function, oscillates with decreasing amplitude.
    fn bessel_j0(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute Bessel function of the first kind, order 1.
    ///
    /// J₁(0) = 0, odd function, oscillates with decreasing amplitude.
    fn bessel_j1(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute Bessel function of the second kind, order 0 (Neumann function).
    ///
    /// Y₀(x) → -∞ as x → 0⁺. Domain: x > 0.
    fn bessel_y0(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute Bessel function of the second kind, order 1 (Neumann function).
    ///
    /// Y₁(x) → -∞ as x → 0⁺. Domain: x > 0.
    fn bessel_y1(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute modified Bessel function of the first kind, order 0.
    ///
    /// I₀(0) = 1, even function, grows exponentially.
    fn bessel_i0(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute modified Bessel function of the first kind, order 1.
    ///
    /// I₁(0) = 0, odd function, grows exponentially.
    fn bessel_i1(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute modified Bessel function of the second kind, order 0.
    ///
    /// K₀(x) → ∞ as x → 0⁺. Domain: x > 0. Decays exponentially.
    fn bessel_k0(&self, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute modified Bessel function of the second kind, order 1.
    ///
    /// K₁(x) → ∞ as x → 0⁺. Domain: x > 0. Decays exponentially.
    fn bessel_k1(&self, x: &Tensor<R>) -> Result<Tensor<R>>;
}

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validate that dtype is suitable for special functions.
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
pub const TWO_OVER_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

/// Euler-Mascheroni constant: γ ≈ 0.5772156649015329
pub const EULER_MASCHERONI: f64 = 0.5772156649015328606065120900824024310422;

/// ln(√(2π)) ≈ 0.9189385332046727 (used in Stirling's approximation)
pub const LN_SQRT_2PI: f64 = 0.9189385332046727417803297364056176398614;

// ============================================================================
// Lanczos Coefficients for Gamma Function
// ============================================================================

/// Lanczos approximation coefficients (g=7, n=9).
pub const LANCZOS_G: f64 = 7.0;

/// Lanczos coefficients for g=7.
pub const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];
