//! Fast Walsh-Hadamard transform trait.

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Walsh-Hadamard transform operations.
pub trait FwhtOps<R: Runtime> {
    /// Normalized Walsh-Hadamard transform on each `block_size` segment of the last axis.
    ///
    /// Uses Sylvester ordering, scale `1/sqrt(block_size)`. `block_size` is a power of
    /// two and divides the last dim. `signs` is a 1-D tensor of width equal to the last
    /// dim, same dtype as `x`, multiplied in before the transform. Output has the shape
    /// and dtype of `x`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when `block_size` is zero, not a power of
    /// two, does not divide the last dim, `x` has zero dimensions, or `signs` is not
    /// 1-D with width equal to the last dim. Returns [`Error::DTypeMismatch`] when
    /// `signs` has a different dtype than `x`. Returns [`Error::UnsupportedDType`]
    /// for a non-floating-point `x`.
    fn fwht(
        &self,
        x: &Tensor<R>,
        block_size: usize,
        signs: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>> {
        let _ = (x, block_size, signs);
        Err(Error::NotImplemented {
            feature: "FwhtOps::fwht",
        })
    }
}
