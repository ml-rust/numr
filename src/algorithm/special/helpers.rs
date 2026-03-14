//! Validation helpers for special mathematical functions.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Validate that dtype is suitable for special functions.
pub fn validate_special_dtype(dtype: DType) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 | DType::F16 | DType::BF16 | DType::FP8E4M3 | DType::FP8E5M2 => {
            Ok(())
        }
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "special function",
        }),
    }
}
