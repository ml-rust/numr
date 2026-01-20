//! Type promotion rules for binary operations

use super::DType;

/// Promote two dtypes to a common dtype for binary operations
///
/// Follows NumPy-like promotion rules:
/// - Floats always win over integers
/// - Larger types win over smaller types
/// - Signed wins over unsigned when mixing
pub fn promote(lhs: DType, rhs: DType) -> DType {
    use DType::*;

    if lhs == rhs {
        return lhs;
    }

    // Promotion priority (higher = wins)
    // FP8 types have low priority since they're storage formats
    // that should promote to higher precision for computation
    // Complex types have highest priority
    let priority = |dt: DType| -> u8 {
        match dt {
            Complex128 => 120,
            Complex64 => 110,
            F64 => 100,
            F32 => 90,
            BF16 => 85,
            F16 => 80,
            FP8E4M3 => 75, // Higher precision FP8 (3 mantissa bits)
            FP8E5M2 => 70, // Lower precision FP8 (2 mantissa bits)
            I64 => 65,
            U64 => 60,
            I32 => 55,
            U32 => 50,
            I16 => 45,
            U16 => 40,
            I8 => 35,
            U8 => 30,
            Bool => 25,
        }
    };

    // Special case: mixing signed and unsigned integers
    // Promote to signed type of same or larger size
    if lhs.is_signed_int() && rhs.is_unsigned_int() {
        return match (lhs, rhs) {
            (I64, _) => I64,
            (I32, U64) => I64,
            (I32, U32) => I64,
            (I32, _) => I32,
            (I16, U64 | U32 | U16) => I32,
            (I16, _) => I16,
            (I8, U64 | U32 | U16 | U8) => I16,
            _ => I64,
        };
    }
    if rhs.is_signed_int() && lhs.is_unsigned_int() {
        return promote(rhs, lhs);
    }

    // General case: higher priority wins
    if priority(lhs) >= priority(rhs) {
        lhs
    } else {
        rhs
    }
}

/// Check if a dtype can be safely cast to another without data loss
///
/// Currently used primarily for validation logic. May be exported in future versions.
#[allow(dead_code)]
pub fn can_cast_safely(from: DType, to: DType) -> bool {
    use DType::*;

    if from == to {
        return true;
    }

    match (from, to) {
        // FP8 can safely cast to wider floats
        (FP8E4M3 | FP8E5M2, F16 | BF16 | F32 | F64) => true,

        // Floats can always accept wider floats
        (F16 | BF16, F32 | F64) => true,
        (F32, F64) => true,

        // Complex widening
        (Complex64, Complex128) => true,

        // Real floats to complex (imaginary part = 0)
        (F32 | F16 | BF16 | FP8E4M3 | FP8E5M2, Complex64) => true,
        (F64 | F32 | F16 | BF16 | FP8E4M3 | FP8E5M2, Complex128) => true,

        // Integers to floats (may lose precision for large values)
        (I8 | U8 | I16 | U16, F32 | F64) => true,
        (I32 | U32, F64) => true,

        // Integers to complex (imaginary part = 0)
        (I8 | U8 | I16 | U16 | I32 | U32 | I64 | U64, Complex64 | Complex128) => true,

        // Integer widening
        (I8, I16 | I32 | I64) => true,
        (I16, I32 | I64) => true,
        (I32, I64) => true,
        (U8, U16 | U32 | U64 | I16 | I32 | I64) => true,
        (U16, U32 | U64 | I32 | I64) => true,
        (U32, U64 | I64) => true,

        // Bool to anything numeric
        (Bool, _) if to.is_int() || to.is_float() || to.is_complex() => true,

        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use DType::*;

    #[test]
    fn test_same_type_promotion() {
        assert_eq!(promote(F32, F32), F32);
        assert_eq!(promote(I64, I64), I64);
    }

    #[test]
    fn test_float_promotion() {
        assert_eq!(promote(F32, F64), F64);
        assert_eq!(promote(F16, F32), F32);
        assert_eq!(promote(BF16, F32), F32);
    }

    #[test]
    fn test_int_float_promotion() {
        // Float always wins
        assert_eq!(promote(I64, F32), F32);
        assert_eq!(promote(I32, F64), F64);
    }

    #[test]
    fn test_signed_unsigned_promotion() {
        assert_eq!(promote(I32, U32), I64);
        assert_eq!(promote(I16, U16), I32);
        assert_eq!(promote(I8, U8), I16);
    }

    #[test]
    fn test_safe_cast() {
        assert!(can_cast_safely(I32, I64));
        assert!(can_cast_safely(F32, F64));
        assert!(can_cast_safely(U8, I32));
        assert!(!can_cast_safely(I64, I32));
        assert!(!can_cast_safely(F64, F32));
    }

    // ========== FP8 Promotion Tests ==========

    #[test]
    fn test_fp8_same_type_promotion() {
        assert_eq!(promote(FP8E4M3, FP8E4M3), FP8E4M3);
        assert_eq!(promote(FP8E5M2, FP8E5M2), FP8E5M2);
    }

    #[test]
    fn test_fp8_to_wider_float_promotion() {
        // FP8 should promote to wider floats
        assert_eq!(promote(FP8E4M3, F16), F16);
        assert_eq!(promote(FP8E4M3, BF16), BF16);
        assert_eq!(promote(FP8E4M3, F32), F32);
        assert_eq!(promote(FP8E4M3, F64), F64);

        assert_eq!(promote(FP8E5M2, F16), F16);
        assert_eq!(promote(FP8E5M2, BF16), BF16);
        assert_eq!(promote(FP8E5M2, F32), F32);
        assert_eq!(promote(FP8E5M2, F64), F64);
    }

    #[test]
    fn test_fp8_e4m3_vs_e5m2_promotion() {
        // E4M3 has higher priority (more mantissa bits = higher precision)
        assert_eq!(promote(FP8E4M3, FP8E5M2), FP8E4M3);
        assert_eq!(promote(FP8E5M2, FP8E4M3), FP8E4M3);
    }

    #[test]
    fn test_fp8_vs_int_promotion() {
        // FP8 (as float) should win over integers
        assert_eq!(promote(FP8E4M3, I64), FP8E4M3);
        assert_eq!(promote(FP8E5M2, I32), FP8E5M2);
    }

    #[test]
    fn test_fp8_safe_cast() {
        // FP8 can safely cast to wider floats
        assert!(can_cast_safely(FP8E4M3, F16));
        assert!(can_cast_safely(FP8E4M3, BF16));
        assert!(can_cast_safely(FP8E4M3, F32));
        assert!(can_cast_safely(FP8E4M3, F64));

        assert!(can_cast_safely(FP8E5M2, F16));
        assert!(can_cast_safely(FP8E5M2, F32));
        assert!(can_cast_safely(FP8E5M2, F64));

        // Cannot safely cast wider floats to FP8 (data loss)
        assert!(!can_cast_safely(F32, FP8E4M3));
        assert!(!can_cast_safely(F16, FP8E5M2));
    }
}
