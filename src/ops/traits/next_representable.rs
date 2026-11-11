//! Adjacent IEEE binary32 and binary64 encodings without floating arithmetic.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Step to the adjacent representable floating-point value.
///
/// CPU and CUDA support F32/F64. WGPU supports F32. Other dtypes return an error.
/// Outputs retain shape, dtype, and device, with independent storage.
/// Strided views and empty tensors follow the same contract.
/// CUDA noncontiguous views support at most eight dimensions. Contiguous rank is unrestricted.
///
/// Integer bit operations preserve NaN signs, payloads, and signaling bits.
/// Subnormal encodings survive regardless of rounding, FTZ, or DAZ modes.
/// These operations do not establish guarantees for surrounding floating arithmetic.
pub trait NextRepresentableOps<R: Runtime> {
    /// Return the immediate greater value for every element.
    ///
    /// Either signed zero becomes the positive minimum subnormal.
    /// Positive infinity remains unchanged. Negative infinity becomes the negative finite maximum.
    /// NaN retains every original bit.
    fn next_up(&self, a: &Tensor<R>) -> Result<Tensor<R>>;

    /// Return the immediate lesser value for every element.
    ///
    /// Either signed zero becomes the negative minimum subnormal.
    /// Negative infinity remains unchanged. Positive infinity becomes the positive finite maximum.
    /// NaN retains every original bit.
    fn next_down(&self, a: &Tensor<R>) -> Result<Tensor<R>>;
}
