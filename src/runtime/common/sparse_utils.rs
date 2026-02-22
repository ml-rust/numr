//! Shared sparse matrix utilities
//!
//! This module provides backend-agnostic utilities for sparse matrix operations.
//! Functions here are used by both CPU and GPU backends to ensure consistency.

use crate::dtype::Element;

/// Compute zero tolerance threshold for sparse operations
///
/// Returns the threshold below which values are considered zero and eliminated
/// from sparse representations. The threshold is dtype-dependent to account for
/// different numeric precision levels.
///
/// # Rationale
///
/// Sparse matrices rely on the assumption that most values are exactly zero.
/// However, due to floating-point rounding errors, operations like subtraction
/// can produce values like `1e-16` instead of exactly `0.0`. These "near-zeros"
/// should be eliminated to:
/// 1. Maintain sparsity (avoid memory bloat)
/// 2. Preserve numerical stability (avoid accumulation of rounding errors)
/// 3. Match user expectations (A - A should be truly sparse)
///
/// # Precision Levels
///
/// - **F64/I64/U64**: `1e-15` - Near machine epsilon, preserves maximum precision
/// - **F32/I32/U32**: `1e-7` - ~7 ULPs (unit of least precision), balances precision and sparsity
/// - **F16/BF16/I16/U16**: `1e-3` - Aggressive threshold due to limited precision
/// - **FP8/I8/U8**: `1e-2` - Very aggressive due to extreme quantization
///
/// # Backend Consistency
///
/// This function is used identically by all backends (CPU, CUDA, WebGPU) to ensure
/// that sparse operations produce the same sparsity pattern regardless of where
/// they execute. Same input tensors always produce same output structure.
pub fn zero_tolerance<T: Element>() -> f64 {
    use std::mem::size_of;
    match size_of::<T>() {
        8 => 1e-15, // F64, I64, U64: ~machine epsilon
        4 => 1e-7,  // F32, I32, U32: ~7 ULPs
        2 => 1e-3,  // F16, BF16, I16, U16: aggressive due to limited precision
        1 => 1e-2,  // FP8, I8, U8: very aggressive due to extreme quantization
        _ => 1e-15, // Default: fallback to highest precision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_tolerance_values() {
        // 8-byte types
        assert_eq!(zero_tolerance::<f64>(), 1e-15);
        assert_eq!(zero_tolerance::<i64>(), 1e-15);
        assert_eq!(zero_tolerance::<u64>(), 1e-15);

        // 4-byte types
        assert_eq!(zero_tolerance::<f32>(), 1e-7);
        assert_eq!(zero_tolerance::<i32>(), 1e-7);
        assert_eq!(zero_tolerance::<u32>(), 1e-7);

        // 2-byte types would be tested if we had f16/bf16 types available

        // 1-byte types
        assert_eq!(zero_tolerance::<i8>(), 1e-2);
        assert_eq!(zero_tolerance::<u8>(), 1e-2);
    }

    #[test]
    fn test_zero_tolerance_filters_correctly() {
        // F32: should filter 1e-8 but keep 1e-6
        let threshold_f32 = zero_tolerance::<f32>();
        assert!(1e-8_f64 < threshold_f32); // Filtered
        assert!(1e-6_f64 > threshold_f32); // Kept

        // F64: should filter 1e-16 but keep 1e-14
        let threshold_f64 = zero_tolerance::<f64>();
        assert!(1e-16_f64 < threshold_f64); // Filtered
        assert!(1e-14_f64 > threshold_f64); // Kept
    }
}
