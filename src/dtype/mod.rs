//! Data type system for numr tensors
//!
//! This module provides the `DType` enum representing all supported element types,
//! along with type promotion rules and conversion utilities.

mod element;
mod promotion;

pub use element::Element;
pub use promotion::promote;

use std::fmt;

/// Data types supported by numr tensors
///
/// This enum represents the element type of a tensor at runtime.
/// Using an enum (rather than generics) allows:
/// - Mixed-precision operations
/// - Runtime type selection
/// - Support for quantized types that aren't `Copy`
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    // Floating point types
    /// 64-bit floating point
    F64 = 0,
    /// 32-bit floating point (most common)
    F32 = 1,
    /// 16-bit floating point (IEEE 754)
    F16 = 2,
    /// 16-bit brain floating point
    BF16 = 3,

    // Integer types
    /// 64-bit signed integer
    I64 = 10,
    /// 32-bit signed integer
    I32 = 11,
    /// 16-bit signed integer
    I16 = 12,
    /// 8-bit signed integer
    I8 = 13,

    // Unsigned integer types
    /// 64-bit unsigned integer
    U64 = 20,
    /// 32-bit unsigned integer
    U32 = 21,
    /// 16-bit unsigned integer
    U16 = 22,
    /// 8-bit unsigned integer
    U8 = 23,

    /// Boolean type
    Bool = 30,
}

impl DType {
    /// Size of one element in bytes
    #[inline]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    /// Returns true if this is a floating point type
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F64 | Self::F32 | Self::F16 | Self::BF16)
    }

    /// Returns true if this is a signed integer type
    #[inline]
    pub const fn is_signed_int(self) -> bool {
        matches!(self, Self::I64 | Self::I32 | Self::I16 | Self::I8)
    }

    /// Returns true if this is an unsigned integer type
    #[inline]
    pub const fn is_unsigned_int(self) -> bool {
        matches!(self, Self::U64 | Self::U32 | Self::U16 | Self::U8)
    }

    /// Returns true if this is any integer type (signed or unsigned)
    #[inline]
    pub const fn is_int(self) -> bool {
        self.is_signed_int() || self.is_unsigned_int()
    }

    /// Returns true if this is a boolean type
    #[inline]
    pub const fn is_bool(self) -> bool {
        matches!(self, Self::Bool)
    }

    /// Returns true if this type can represent negative values
    #[inline]
    pub const fn is_signed(self) -> bool {
        self.is_float() || self.is_signed_int()
    }

    /// Get the default dtype for floating point operations
    #[inline]
    pub const fn default_float() -> Self {
        Self::F32
    }

    /// Get the default dtype for integer operations
    #[inline]
    pub const fn default_int() -> Self {
        Self::I64
    }

    /// Short name for display (e.g., "f32", "i64")
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::I64 => "i64",
            Self::I32 => "i32",
            Self::I16 => "i16",
            Self::I8 => "i8",
            Self::U64 => "u64",
            Self::U32 => "u32",
            Self::U16 => "u16",
            Self::U8 => "u8",
            Self::Bool => "bool",
        }
    }

    /// Minimum value representable by this dtype (as f64)
    pub fn min_value(self) -> f64 {
        match self {
            Self::F64 => f64::MIN,
            Self::F32 => f32::MIN as f64,
            Self::F16 => -65504.0, // IEEE 754 half precision
            Self::BF16 => -3.4e38, // Approximate
            Self::I64 => i64::MIN as f64,
            Self::I32 => i32::MIN as f64,
            Self::I16 => i16::MIN as f64,
            Self::I8 => i8::MIN as f64,
            Self::U64 => 0.0,
            Self::U32 => 0.0,
            Self::U16 => 0.0,
            Self::U8 => 0.0,
            Self::Bool => 0.0,
        }
    }

    /// Maximum value representable by this dtype (as f64)
    pub fn max_value(self) -> f64 {
        match self {
            Self::F64 => f64::MAX,
            Self::F32 => f32::MAX as f64,
            Self::F16 => 65504.0,
            Self::BF16 => 3.4e38,
            Self::I64 => i64::MAX as f64,
            Self::I32 => i32::MAX as f64,
            Self::I16 => i16::MAX as f64,
            Self::I8 => i8::MAX as f64,
            Self::U64 => u64::MAX as f64,
            Self::U32 => u32::MAX as f64,
            Self::U16 => u16::MAX as f64,
            Self::U8 => u8::MAX as f64,
            Self::Bool => 1.0,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_name())
    }
}

/// Set of dtypes for efficient membership testing
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DTypeSet {
    bits: u64,
}

impl DTypeSet {
    /// Empty set
    pub const EMPTY: Self = Self { bits: 0 };

    /// All floating point types
    pub const FLOATS: Self = Self {
        bits: (1 << DType::F64 as u8)
            | (1 << DType::F32 as u8)
            | (1 << DType::F16 as u8)
            | (1 << DType::BF16 as u8),
    };

    /// All signed integer types
    pub const SIGNED_INTS: Self = Self {
        bits: (1 << DType::I64 as u8)
            | (1 << DType::I32 as u8)
            | (1 << DType::I16 as u8)
            | (1 << DType::I8 as u8),
    };

    /// All unsigned integer types
    pub const UNSIGNED_INTS: Self = Self {
        bits: (1 << DType::U64 as u8)
            | (1 << DType::U32 as u8)
            | (1 << DType::U16 as u8)
            | (1 << DType::U8 as u8),
    };

    /// All integer types
    pub const INTS: Self = Self {
        bits: Self::SIGNED_INTS.bits | Self::UNSIGNED_INTS.bits,
    };

    /// All numeric types (floats + ints)
    pub const NUMERIC: Self = Self {
        bits: Self::FLOATS.bits | Self::INTS.bits,
    };

    /// Create a set containing a single dtype
    #[inline]
    pub const fn single(dtype: DType) -> Self {
        Self {
            bits: 1 << dtype as u8,
        }
    }

    /// Check if the set contains a dtype
    #[inline]
    pub const fn contains(self, dtype: DType) -> bool {
        self.bits & (1 << dtype as u8) != 0
    }

    /// Union of two sets
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
        }
    }

    /// Intersection of two sets
    #[inline]
    pub const fn intersection(self, other: Self) -> Self {
        Self {
            bits: self.bits & other.bits,
        }
    }

    /// Check if set is empty
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F64.size_in_bytes(), 8);
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert_eq!(DType::I8.size_in_bytes(), 1);
        assert_eq!(DType::Bool.size_in_bytes(), 1);
    }

    #[test]
    fn test_dtype_categories() {
        assert!(DType::F32.is_float());
        assert!(!DType::I32.is_float());
        assert!(DType::I32.is_signed_int());
        assert!(DType::U32.is_unsigned_int());
        assert!(!DType::U32.is_signed());
    }

    #[test]
    fn test_dtype_set() {
        assert!(DTypeSet::FLOATS.contains(DType::F32));
        assert!(!DTypeSet::FLOATS.contains(DType::I32));
        assert!(DTypeSet::INTS.contains(DType::I32));
        assert!(DTypeSet::NUMERIC.contains(DType::F32));
        assert!(DTypeSet::NUMERIC.contains(DType::I32));
    }
}
