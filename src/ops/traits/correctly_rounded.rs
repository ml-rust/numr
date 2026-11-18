//! Correctly rounded F64 elementary functions on tensors.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::impl_generic::interval::{require_basic, require_fused};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Elementwise F64 elementary functions correctly rounded to nearest, ties to even.
///
/// Each method maps one function of [`crate::algorithm::correctly_rounded`] over the elements.
///
/// - Input is F64 on the client device, in any layout. Output is a fresh contiguous F64
///   tensor of the same shape. An empty input gives an empty output.
/// - `cr_exp`, `cr_log` and `cr_log1p`: each element is the infinitely precise result
///   rounded to nearest, ties to even, with gradual underflow. Overflow gives `+inf`.
/// - `cr_pow_scalar`: the same for every input pair whose final 256-bit rounding test
///   succeeds. CORE-MATH proves correct rounding of `pow` modulo the hardest-to-round
///   cases, which are unknown for binary64. No known input fails the test. If one does,
///   the element is the 256-bit approximation rounded to nearest.
/// - Every runtime that implements the trait returns identical bits for identical input
///   bits. The host functions return the same bits. The bits do not depend on the tensor
///   length, layout, thread count or platform libm.
/// - Every NaN result is the canonical quiet NaN `0x7ff8000000000000`. No input NaN
///   payload survives. IEEE exception flags are not promised.
///
/// The runtime must declare [`crate::runtime::FloatArithmeticContract::Ieee754Basic`] and
/// [`crate::runtime::FusedMulAddContract::Ieee754Fused`] for F64. The caller keeps
/// round-to-nearest-even and gradual underflow active. The op does not detect or change
/// that environment.
///
/// The trait sits outside [`crate::ops::TensorOps`], so an external runtime need not
/// implement it. An implementation calls [`require_correctly_rounded_contracts`] first.
///
/// # Errors
///
/// - [`crate::error::Error::UnsupportedDType`] for any dtype other than F64.
/// - [`crate::error::Error::DeviceMismatch`] for a tensor on another device.
/// - The [`require_correctly_rounded_contracts`] errors when a contract is missing.
/// - [`crate::error::Error::BackendLimitation`] on a runtime that computes no F64, such as
///   WebGPU.
pub trait CorrectlyRoundedOps<R: Runtime> {
    /// `e^a`, correctly rounded. See [`crate::algorithm::correctly_rounded::exp`].
    fn cr_exp(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Natural logarithm of `a`, correctly rounded. See
    /// [`crate::algorithm::correctly_rounded::log`].
    fn cr_log(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// `ln(1 + a)`, correctly rounded, without forming `1 + a`. See
    /// [`crate::algorithm::correctly_rounded::log1p`].
    fn cr_log1p(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// `a^exponent` with a host exponent and C Annex F special values. See
    /// [`crate::algorithm::correctly_rounded::pow`] for the rounding guarantee.
    fn cr_pow_scalar(&self, a: &Tensor<R>, exponent: f64) -> Result<Tensor<R>>;
}

/// Reject a runtime that lacks the F64 arithmetic contracts [`CorrectlyRoundedOps`] needs.
///
/// # Errors
///
/// - [`crate::error::Error::BackendLimitation`] naming `op` when the runtime does not
///   declare [`crate::runtime::FloatArithmeticContract::Ieee754Basic`] for F64.
/// - [`crate::error::Error::InvalidArgument`] naming `op` when the runtime does not
///   declare [`crate::runtime::FusedMulAddContract::Ieee754Fused`] for F64.
pub fn require_correctly_rounded_contracts<R: Runtime<DType = DType>>(
    op: &'static str,
) -> Result<()> {
    require_basic::<R>(DType::F64, op)?;
    require_fused::<R>(DType::F64, op)
}
