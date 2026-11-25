//! Borrowed operands and owned results of interval and expansion kernels.

use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Borrowed endpoints of one interval operand.
///
/// Both endpoints require the same shape, dtype, and device.
#[derive(Debug)]
pub struct IntervalOperand<'a, R: Runtime> {
    /// Lower endpoints.
    pub lower: &'a Tensor<R>,
    /// Upper endpoints.
    pub upper: &'a Tensor<R>,
}

impl<R: Runtime> Clone for IntervalOperand<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime> Copy for IntervalOperand<'_, R> {}

/// Borrowed components of one expansion `[high + low - abs_error, high + low + abs_error]`.
///
/// All components require the same shape, dtype, and device.
#[derive(Debug)]
pub struct ExpansionOperand<'a, R: Runtime> {
    /// High center component.
    pub high: &'a Tensor<R>,
    /// Low center component.
    pub low: &'a Tensor<R>,
    /// Absolute error radius.
    pub abs_error: &'a Tensor<R>,
}

impl<R: Runtime> Clone for ExpansionOperand<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime> Copy for ExpansionOperand<'_, R> {}

/// Owned `(lower, upper)` interval endpoints.
pub type IntervalPair<R> = (Tensor<R>, Tensor<R>);

/// Owned `(high, low, abs_error)` expansion components.
pub type ExpansionTriple<R> = (Tensor<R>, Tensor<R>, Tensor<R>);

/// Enclosure rule for a binary interval operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntervalEnclosure {
    /// Outward rounding with exact-zero and exact-one identities only.
    Outward,
    /// Outward rounding, then exact singleton retention, then sign restriction.
    ///
    /// Addition and subtraction retain singleton sums that TwoSum proves exact.
    /// Multiplication and positive division retain singleton results that a fused residual proves exact.
    /// That retention applies only under [`crate::runtime::FusedMulAddContract::Ieee754Fused`].
    /// Multiplication and positive division then clamp bounds to signs that operand endpoints prove.
    Tightened,
}
