//! Shared validation logic for complex number operations.
//!
//! These functions are used across CPU, CUDA, and WebGPU backends
//! to validate inputs for complex operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Validate that real and imag tensors have matching shapes and dtypes for make_complex.
///
/// # Supported dtypes
///
/// - F32 → Complex64
/// - F64 → Complex128 (CPU and CUDA only, not WebGPU)
///
/// # Errors
///
/// - `ShapeMismatch` if real and imag have different shapes
/// - `DTypeMismatch` if real and imag have different dtypes
/// - `UnsupportedDType` if dtype is not F32 or F64
pub fn validate_make_complex_inputs<R: Runtime>(real: &Tensor<R>, imag: &Tensor<R>) -> Result<()> {
    // Check shapes match
    if real.shape() != imag.shape() {
        return Err(Error::ShapeMismatch {
            expected: real.shape().to_vec(),
            got: imag.shape().to_vec(),
        });
    }

    // Check dtypes match
    if real.dtype() != imag.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: real.dtype(),
            rhs: imag.dtype(),
        });
    }

    // Check dtype is supported (F32 or F64)
    match real.dtype() {
        DType::F32 | DType::F64 => Ok(()),
        dtype => Err(Error::UnsupportedDType {
            dtype,
            op: "make_complex",
        }),
    }
}

/// Validate that real and imag tensors are F32 only (for WebGPU backend).
///
/// WebGPU does not support F64/Complex128, so this is a stricter validation.
///
/// # Errors
///
/// - `ShapeMismatch` if real and imag have different shapes
/// - `DTypeMismatch` if real and imag have different dtypes
/// - `UnsupportedDType` if dtype is not F32
#[cfg(feature = "wgpu")]
pub fn validate_make_complex_inputs_f32_only<R: Runtime>(
    real: &Tensor<R>,
    imag: &Tensor<R>,
) -> Result<()> {
    // Check shapes match
    if real.shape() != imag.shape() {
        return Err(Error::ShapeMismatch {
            expected: real.shape().to_vec(),
            got: imag.shape().to_vec(),
        });
    }

    // Check dtypes match
    if real.dtype() != imag.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: real.dtype(),
            rhs: imag.dtype(),
        });
    }

    // Check dtype is F32 only (WebGPU doesn't support F64)
    match real.dtype() {
        DType::F32 => Ok(()),
        DType::F64 => Err(Error::UnsupportedDType {
            dtype: DType::F64,
            op: "make_complex (WebGPU does not support F64)",
        }),
        dtype => Err(Error::UnsupportedDType {
            dtype,
            op: "make_complex",
        }),
    }
}

/// Validate inputs for complex × real operations (CPU and CUDA).
///
/// # Valid combinations
///
/// - Complex64 × F32 → Complex64
/// - Complex128 × F64 → Complex128
///
/// # Errors
///
/// - `ShapeMismatch` if shapes don't match
/// - `DTypeMismatch` if real dtype doesn't match complex component dtype
/// - `UnsupportedDType` if complex is not Complex64/Complex128
pub fn validate_complex_real_inputs<R: Runtime>(
    complex: &Tensor<R>,
    real: &Tensor<R>,
    op: &'static str,
) -> Result<()> {
    // Check shapes match
    if complex.shape() != real.shape() {
        return Err(Error::ShapeMismatch {
            expected: complex.shape().to_vec(),
            got: real.shape().to_vec(),
        });
    }

    // Check dtypes are compatible
    match (complex.dtype(), real.dtype()) {
        (DType::Complex64, DType::F32) => Ok(()),
        (DType::Complex128, DType::F64) => Ok(()),
        (DType::Complex64, other) => Err(Error::DTypeMismatch {
            lhs: DType::Complex64,
            rhs: other,
        }),
        (DType::Complex128, other) => Err(Error::DTypeMismatch {
            lhs: DType::Complex128,
            rhs: other,
        }),
        (other, _) => Err(Error::UnsupportedDType { dtype: other, op }),
    }
}

/// Validate inputs for complex × real operations (WebGPU only, F32).
///
/// WebGPU only supports Complex64 × F32.
///
/// # Errors
///
/// - `ShapeMismatch` if shapes don't match
/// - `DTypeMismatch` if real dtype is not F32
/// - `UnsupportedDType` if complex is not Complex64 or if Complex128 is used
#[cfg(feature = "wgpu")]
pub fn validate_complex_real_inputs_f32_only<R: Runtime>(
    complex: &Tensor<R>,
    real: &Tensor<R>,
    op: &'static str,
) -> Result<()> {
    // Check shapes match
    if complex.shape() != real.shape() {
        return Err(Error::ShapeMismatch {
            expected: complex.shape().to_vec(),
            got: real.shape().to_vec(),
        });
    }

    // Check dtypes are compatible (WebGPU only supports Complex64 × F32)
    match (complex.dtype(), real.dtype()) {
        (DType::Complex64, DType::F32) => Ok(()),
        (DType::Complex64, other) => Err(Error::DTypeMismatch {
            lhs: DType::F32,
            rhs: other,
        }),
        (DType::Complex128, _) => Err(Error::UnsupportedDType {
            dtype: DType::Complex128,
            op,
        }),
        (other, _) => Err(Error::UnsupportedDType { dtype: other, op }),
    }
}
