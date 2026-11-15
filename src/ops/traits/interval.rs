//! Combined outward interval kernels.

use super::interval_operand::{IntervalEnclosure, IntervalOperand, IntervalPair};
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Elementwise interval arithmetic that runs each operation as one pass.
///
/// Results equal the documented composition of ordinary rounded tensor operations bit for bit.
/// That equality covers NaN payloads, signed zeros, infinities, and subnormals.
/// F32 and F64 only. WGPU supports F32.
///
/// # Status flags
///
/// Every operation sets a shared status flag instead of reducing a mask per operation.
/// Create the flag with [`IntervalOps::interval_status`]. Read it once with `item::<u32>()`.
/// A nonzero value means some lane produced a nonfinite or inverted bound, or left a domain.
/// The flag only ever changes from zero to one.
/// Interval and expansion results remain provisional until the caller reads a zero flag.
/// Flags accumulate whether or not an operation later returns an error.
///
/// Interval operands broadcast. Expansion components require identical shapes.
pub trait IntervalOps<R: Runtime> {
    /// Create a cleared U32 scalar status flag on this client's device.
    fn interval_status(&self) -> Result<Tensor<R>>;

    /// Probe the arithmetic these kernels run for `dtype` and return a U32 scalar of failure bits.
    ///
    /// Bits, from least significant: positive ties to even, negative ties to even,
    /// gradual subnormal inputs, gradual multiplication underflow, gradual division underflow,
    /// and the positive, negative, and cancellation TwoSum witnesses.
    /// Every probe runs on the device with runtime operands, so no host constant folding applies.
    fn interval_envelope(&self, dtype: DType) -> Result<Tensor<R>>;

    /// Set `status` when any element of `mask` is nonzero. NaN counts as nonzero.
    ///
    /// Supports F32, F64, I64, I32, U32, and U8 masks.
    fn interval_flag_nonzero(&self, mask: &Tensor<R>, status: &Tensor<R>) -> Result<()>;

    /// Set `status` when any element of `value` is NaN or infinite.
    fn interval_flag_nonfinite(&self, value: &Tensor<R>, status: &Tensor<R>) -> Result<()>;

    /// Set `status` when any lane has a nonfinite endpoint or `lower > upper`.
    fn interval_flag_bounds(&self, value: IntervalOperand<'_, R>, status: &Tensor<R>)
    -> Result<()>;

    /// Copy both endpoints into fresh storage and set `status` for invalid bounds, as above.
    ///
    /// When `lower` and `upper` are the same view, one copy serves both endpoints.
    fn interval_capture(
        &self,
        value: IntervalOperand<'_, R>,
        status: &Tensor<R>,
    ) -> Result<IntervalPair<R>>;

    /// Enclose `a + b`: `[next_down(a.lower + b.lower), next_up(a.upper + b.upper)]`.
    ///
    /// An exact-zero operand `[0, 0]` returns the other operand's endpoint bits.
    /// Sets `status` when a result bound is nonfinite or inverted.
    fn interval_add(
        &self,
        a: IntervalOperand<'_, R>,
        b: IntervalOperand<'_, R>,
        enclosure: IntervalEnclosure,
        status: &Tensor<R>,
    ) -> Result<IntervalPair<R>>;

    /// Enclose `a - b`: `[next_down(a.lower - b.upper), next_up(a.upper - b.lower)]`.
    ///
    /// An exact-zero `b` returns `a`. An exact-zero `a` returns the negated, swapped endpoints of `b`.
    /// Sets `status` when a result bound is nonfinite or inverted.
    fn interval_sub(
        &self,
        a: IntervalOperand<'_, R>,
        b: IntervalOperand<'_, R>,
        enclosure: IntervalEnclosure,
        status: &Tensor<R>,
    ) -> Result<IntervalPair<R>>;

    /// Enclose `a * b` over all four endpoint products, then step each extremum outward.
    ///
    /// An exact-zero operand returns that operand's endpoint bits.
    /// Zero lanes never evaluate zero times infinity.
    /// Sets `status` when a result bound is nonfinite or inverted.
    fn interval_mul(
        &self,
        a: IntervalOperand<'_, R>,
        b: IntervalOperand<'_, R>,
        enclosure: IntervalEnclosure,
        status: &Tensor<R>,
    ) -> Result<IntervalPair<R>>;

    /// Enclose `a / divisor` for represented divisors that must be finite and strictly positive.
    ///
    /// Sets `status` for nonfinite or nonpositive divisors, checked on the divisor's own shape.
    /// Invalid divisor lanes divide by one. A divisor of exactly one returns `a` unchanged.
    /// An exact-zero `a` returns `a` unchanged.
    fn interval_div_positive(
        &self,
        a: IntervalOperand<'_, R>,
        divisor: &Tensor<R>,
        enclosure: IntervalEnclosure,
        status: &Tensor<R>,
    ) -> Result<IntervalPair<R>>;

    /// Enclose `a / b` for intervals whose finite lower endpoints must be strictly positive.
    ///
    /// Sets `status` for nonfinite or inverted `a`, and for nonfinite, inverted, or nonpositive `b`.
    /// Each operand is checked on its own shape. Invalid lanes evaluate finite stand-in operands.
    /// A divisor of exactly `[1, 1]` or an exact-zero `a` returns `a` unchanged.
    fn interval_div_positive_interval(
        &self,
        a: IntervalOperand<'_, R>,
        b: IntervalOperand<'_, R>,
        enclosure: IntervalEnclosure,
        status: &Tensor<R>,
    ) -> Result<IntervalPair<R>>;

    /// Enclose square roots of finite nonnegative ordered intervals.
    ///
    /// Sets `status` for nonfinite, inverted, or negative-lower inputs, which evaluate zero instead.
    /// Zero endpoints keep their bits. The square root uses the backend's rounded `sqrt`.
    fn interval_sqrt_nonnegative(
        &self,
        a: IntervalOperand<'_, R>,
        status: &Tensor<R>,
    ) -> Result<IntervalPair<R>>;
}
