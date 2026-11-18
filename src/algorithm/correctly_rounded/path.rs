//! Which branch of a kernel produced a result.

/// The branch that produced one result.
///
/// The kernels return it beside every value. The public functions drop it, so it costs nothing there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Path {
    /// A special input: NaN, infinity, zero, a tiny input, or an overflow or underflow.
    Special,
    /// The fast path, certified by its rounding test.
    Fast,
    /// The accurate path, run after the fast-path rounding test failed.
    Accurate,
    /// A listed input whose result the accurate path takes from the exception table.
    Exception,
    /// An exact or midpoint `pow` result, found after a rounding test failed.
    Exact,
    /// The 256-bit `pow` path, run after the accurate-path rounding test failed.
    Extended,
    /// The 256-bit `pow` rounding test failed. No known input reaches this branch.
    /// The result is the 256-bit approximation rounded to nearest.
    Uncertified,
}
