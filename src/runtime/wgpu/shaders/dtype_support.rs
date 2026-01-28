//! DType support validation for WebGPU operations
//!
//! This module defines which operations support which dtypes and provides
//! validation functions to ensure operations are called with supported types.

use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Unary Operations Support
// ============================================================================

/// Operations that work for all dtypes (F32, I32, U32)
const UNIVERSAL_UNARY_OPS: &[&str] = &["abs", "square", "sign"];

/// Operations that work for signed types only (F32, I32)
const SIGNED_UNARY_OPS: &[&str] = &["neg"];

/// Operations that require floating point (F32 only)
const FLOAT_ONLY_UNARY_OPS: &[&str] = &[
    "sqrt", "exp", "log", "sin", "cos", "tan", "tanh", "recip", "floor", "ceil", "round", "relu",
    "sigmoid", "silu", "gelu", "isnan", "isinf",
];

/// Check if a unary operation supports the given dtype
pub fn is_unary_op_supported(op: &str, dtype: DType) -> bool {
    // Universal ops work for all types
    if UNIVERSAL_UNARY_OPS.contains(&op) {
        return matches!(dtype, DType::F32 | DType::I32 | DType::U32);
    }

    // Signed ops don't work for U32
    if SIGNED_UNARY_OPS.contains(&op) {
        return matches!(dtype, DType::F32 | DType::I32);
    }

    // Float-only ops
    if FLOAT_ONLY_UNARY_OPS.contains(&op) {
        return dtype == DType::F32;
    }

    // Default: assume F32 only for unknown ops
    dtype == DType::F32
}

/// Validate that a unary operation supports the given dtype
pub fn check_unary_dtype_support(op: &'static str, dtype: DType) -> Result<()> {
    if !is_unary_op_supported(op, dtype) {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

// ============================================================================
// Binary Operations Support
// ============================================================================

/// All binary operations support F32, I32, U32
const BINARY_OPS: &[&str] = &["add", "sub", "mul", "div", "max", "min"];

/// Pow operation (requires special handling for integers)
const POW_OP: &str = "pow";

/// Check if a binary operation supports the given dtype
pub fn is_binary_op_supported(op: &str, dtype: DType) -> bool {
    if BINARY_OPS.contains(&op) || op == POW_OP {
        return matches!(dtype, DType::F32 | DType::I32 | DType::U32);
    }
    // Default: assume F32 only
    dtype == DType::F32
}

/// Validate that a binary operation supports the given dtype
pub fn check_binary_dtype_support(op: &'static str, dtype: DType) -> Result<()> {
    if !is_binary_op_supported(op, dtype) {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

// ============================================================================
// Scalar Operations Support
// ============================================================================

/// All scalar operations support F32, I32, U32
pub fn is_scalar_op_supported(_op: &str, dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::I32 | DType::U32)
}

/// Validate that a scalar operation supports the given dtype
pub fn check_scalar_dtype_support(op: &'static str, dtype: DType) -> Result<()> {
    if !is_scalar_op_supported(op, dtype) {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

// ============================================================================
// Comparison Operations Support
// ============================================================================

/// All comparison operations support F32, I32, U32
pub fn is_compare_op_supported(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::I32 | DType::U32)
}

/// Validate that comparison operations support the given dtype
pub fn check_compare_dtype_support(op: &'static str, dtype: DType) -> Result<()> {
    if !is_compare_op_supported(dtype) {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universal_unary_ops() {
        // abs works for all types
        assert!(is_unary_op_supported("abs", DType::F32));
        assert!(is_unary_op_supported("abs", DType::I32));
        assert!(is_unary_op_supported("abs", DType::U32));

        // square works for all types
        assert!(is_unary_op_supported("square", DType::F32));
        assert!(is_unary_op_supported("square", DType::I32));
        assert!(is_unary_op_supported("square", DType::U32));
    }

    #[test]
    fn test_signed_unary_ops() {
        // neg works for F32 and I32, not U32
        assert!(is_unary_op_supported("neg", DType::F32));
        assert!(is_unary_op_supported("neg", DType::I32));
        assert!(!is_unary_op_supported("neg", DType::U32));
    }

    #[test]
    fn test_float_only_unary_ops() {
        // sqrt is F32 only
        assert!(is_unary_op_supported("sqrt", DType::F32));
        assert!(!is_unary_op_supported("sqrt", DType::I32));
        assert!(!is_unary_op_supported("sqrt", DType::U32));

        // relu is F32 only
        assert!(is_unary_op_supported("relu", DType::F32));
        assert!(!is_unary_op_supported("relu", DType::I32));
        assert!(!is_unary_op_supported("relu", DType::U32));
    }

    #[test]
    fn test_binary_ops_all_dtypes() {
        for &op in &["add", "sub", "mul", "div", "max", "min", "pow"] {
            assert!(is_binary_op_supported(op, DType::F32));
            assert!(is_binary_op_supported(op, DType::I32));
            assert!(is_binary_op_supported(op, DType::U32));
        }
    }

    #[test]
    fn test_scalar_ops_all_dtypes() {
        assert!(is_scalar_op_supported("add_scalar", DType::F32));
        assert!(is_scalar_op_supported("add_scalar", DType::I32));
        assert!(is_scalar_op_supported("add_scalar", DType::U32));
    }

    #[test]
    fn test_compare_ops_all_dtypes() {
        assert!(is_compare_op_supported(DType::F32));
        assert!(is_compare_op_supported(DType::I32));
        assert!(is_compare_op_supported(DType::U32));
    }
}
