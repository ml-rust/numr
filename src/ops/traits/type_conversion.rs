//! Type conversion operations trait.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Type conversion operations
pub trait TypeConversionOps<R: Runtime> {
    /// Cast tensor to a different data type.
    ///
    /// Converts all elements of the input tensor to the target dtype.
    /// The output tensor has the same shape as the input.
    ///
    /// # Supported Conversions
    ///
    /// - **Widening** (lossless): I8→I16→I32→I64, F16→F32→F64
    /// - **Narrowing** (may lose precision): F64→F32→F16, I64→I32→I16→I8
    /// - **Float↔Int**: Truncates toward zero for float→int
    /// - **FP8 conversions**: F32↔FP8E4M3, F32↔FP8E5M2 (with saturation)
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor
    /// * `dtype` - Target data type
    ///
    /// # Returns
    ///
    /// New tensor with the specified dtype
    ///
    /// # Errors
    ///
    /// Returns `UnsupportedDType` if the conversion is not supported
    /// (e.g., Bool↔numeric without explicit handling).
    fn cast(&self, a: &Tensor<R>, dtype: DType) -> Result<Tensor<R>> {
        let _ = (a, dtype);
        Err(Error::NotImplemented {
            feature: "TypeConversionOps::cast",
        })
    }
}
