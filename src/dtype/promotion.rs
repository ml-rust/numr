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
    let priority = |dt: DType| -> u8 {
        match dt {
            F64 => 100,
            F32 => 90,
            BF16 => 85,
            F16 => 80,
            I64 => 70,
            U64 => 65,
            I32 => 60,
            U32 => 55,
            I16 => 50,
            U16 => 45,
            I8 => 40,
            U8 => 35,
            Bool => 30,
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
pub fn can_cast_safely(from: DType, to: DType) -> bool {
    use DType::*;

    if from == to {
        return true;
    }

    match (from, to) {
        // Floats can always accept wider floats
        (F16 | BF16, F32 | F64) => true,
        (F32, F64) => true,

        // Integers to floats (may lose precision for large values)
        (I8 | U8 | I16 | U16, F32 | F64) => true,
        (I32 | U32, F64) => true,

        // Integer widening
        (I8, I16 | I32 | I64) => true,
        (I16, I32 | I64) => true,
        (I32, I64) => true,
        (U8, U16 | U32 | U64 | I16 | I32 | I64) => true,
        (U16, U32 | U64 | I32 | I64) => true,
        (U32, U64 | I64) => true,

        // Bool to anything numeric
        (Bool, _) if to.is_int() || to.is_float() => true,

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
}
