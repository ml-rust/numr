//! WGSL shader generation for special mathematical functions
//!
//! Implements erf, erfc, erfinv, gamma, lgamma, digamma, beta,
//! betainc, gammainc, gammaincc using numerical algorithms in WGSL.
//!
//! # Module Structure
//!
//! - `common` - Shared constants and helper functions
//! - `unary` - Unary function shaders (erf, erfc, erfinv, gamma, lgamma, digamma)
//! - `binary` - Binary function shaders (beta, gammainc, gammaincc)
//! - `ternary` - Ternary function shaders (betainc)

mod binary;
mod ternary;
mod unary;

pub use binary::generate_special_binary_shader;
pub use ternary::generate_special_ternary_shader;
pub use unary::generate_special_unary_shader;

// ============================================================================
// Shared Constants and Helpers
// ============================================================================

/// Generate WGSL constants used by all special function shaders.
pub(super) fn common_constants() -> &'static str {
    r#"const WORKGROUP_SIZE: u32 = 256u;
const PI: f32 = 3.14159265358979323846;
const SQRT_PI: f32 = 1.7724538509055159;
const EULER_GAMMA: f32 = 0.5772156649015329;
const LN_SQRT_2PI: f32 = 0.9189385332046727;
const LANCZOS_G: f32 = 7.0;
const MAX_ITER: i32 = 100;
const EPSILON: f32 = 1e-6;
const TINY: f32 = 1e-30;"#
}

/// Generate the common lgamma helper functions used by multiple shaders.
///
/// These functions are shared between unary, binary, and ternary shaders
/// to avoid code duplication (~50 lines saved per shader).
pub(super) fn lgamma_helpers() -> &'static str {
    r#"
// Lanczos computation for positive x only (no recursion)
fn lgamma_positive(x: f32) -> f32 {
    // Lanczos coefficients (g=7, n=9)
    let c0 = 0.99999999999980993;
    let c1 = 676.5203681218851;
    let c2 = -1259.1392167224028;
    let c3 = 771.32342877765313;
    let c4 = -176.61502916214059;
    let c5 = 12.507343278686905;
    let c6 = -0.13857109526572012;
    let c7 = 9.9843695780195716e-6;
    let c8 = 1.5056327351493116e-7;

    let z = x - 1.0;
    var ag = c0;
    ag = ag + c1 / (z + 1.0);
    ag = ag + c2 / (z + 2.0);
    ag = ag + c3 / (z + 3.0);
    ag = ag + c4 / (z + 4.0);
    ag = ag + c5 / (z + 5.0);
    ag = ag + c6 / (z + 6.0);
    ag = ag + c7 / (z + 7.0);
    ag = ag + c8 / (z + 8.0);

    let t = z + LANCZOS_G + 0.5;
    return LN_SQRT_2PI + (z + 0.5) * log(t) - t + log(ag);
}

// Log-gamma using Lanczos approximation (non-recursive)
fn lgamma_impl(x: f32) -> f32 {
    if (x <= 0.0) {
        // Use reflection formula for negative values
        if (x == floor(x)) {
            return 1e30; // Pole at non-positive integers
        }
        // lgamma(x) = log(pi / sin(pi*x)) - lgamma(1-x)
        // Since 1-x > 0 for x <= 0, we call lgamma_positive directly
        let sinpix = sin(PI * x);
        if (sinpix == 0.0) {
            return 1e30;
        }
        return log(PI / abs(sinpix)) - lgamma_positive(1.0 - x);
    }

    return lgamma_positive(x);
}"#
}
