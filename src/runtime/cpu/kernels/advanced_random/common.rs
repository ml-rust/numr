//! Shared helpers used by every advanced-PRNG kernel.

use std::f64::consts::PI;

/// Box-Muller transform: convert two uniform values to two standard normal values
///
/// This is the shared implementation used by all PRNGs for consistency.
#[inline(always)]
pub(crate) fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    // Clamp to avoid log(0) and ensure valid range
    let u1 = u1.clamp(1e-10, 1.0 - 1e-10);

    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * PI * u2;

    (r * theta.cos(), r * theta.sin())
}
