//! Data type system for numr tensors
//!
//! This module provides the `DType` enum representing all supported element types,
//! along with type promotion rules and conversion utilities.

pub mod complex;
mod element;
pub mod fp8;
mod promotion;

pub use complex::{Complex64, Complex128};
pub use element::Element;
pub use fp8::{FP8E4M3, FP8E5M2};
pub use promotion::promote;

use std::fmt;

// ============================================================================
// Mixed Precision Configuration
// ============================================================================

/// Compute precision for intermediate calculations with reduced-precision types.
///
/// When operating on reduced-precision types (F16, BF16, FP8), values are typically
/// converted to a higher precision format for computation, then converted back.
/// This allows trading off speed vs precision.
///
/// # Precision Comparison
///
/// | Precision | Decimal Digits | Speed   | Use Case |
/// |-----------|----------------|---------|----------|
/// | **F64**   | ~15-16         | Slowest | Scientific computing requiring maximum precision |
/// | **F32**   | ~7             | Medium  | High-precision ML, when BF16 isn't enough |
/// | **BF16**  | ~3             | Fastest | ML training/inference (default, industry standard) |
///
/// # Applicability
///
/// - **FP8**: Always needs upcasting (8-bit storage, compute in BF16, F32, or F64)
/// - **F16/BF16**: Can optionally upcast to F32/F64 for higher precision
/// - **F32**: Can upcast to F64 for scientific computing
/// - **F64**: No upcasting needed (already highest precision)
///
/// # Resolution Order
///
/// `per-operation > tensor-level > client default`
///
/// # Default
///
/// BF16 is the default, as it provides good speed with the same dynamic range as F32.
/// This is the industry standard for mixed-precision ML training.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ComputePrecision {
    /// Compute in F64 (highest precision, slowest)
    /// Use for: scientific simulations, physics, when F32 precision is insufficient
    F64,
    /// Compute in F32 (high precision, medium speed)
    /// Use for: high-precision ML, numerical algorithms sensitive to rounding
    F32,
    /// Compute in BF16 (lower precision, fastest, industry standard for ML)
    /// Use for: ML training/inference, when speed matters more than precision
    #[default]
    BF16,
}

/// Accumulation precision for reductions and matrix multiplications.
///
/// During operations like sum, mean, and matmul, intermediate results are accumulated.
/// Higher precision accumulation prevents numerical drift from repeated additions.
///
/// # Precision Comparison
///
/// | Precision | Decimal Digits | Use Case |
/// |-----------|----------------|----------|
/// | **F64**   | ~15-16         | Scientific computing, physics simulations, CFD |
/// | **F32**   | ~7             | ML/AI training, general computing (default) |
/// | **BF16**  | ~3             | Research, memory-constrained scenarios |
///
/// # When It Matters
///
/// - **Scientific simulations**: F64 required for physics, climate, quantum mechanics
/// - **Gradient accumulation**: F32 usually sufficient for ML
/// - **Matrix multiplication**: Dot product accumulation
/// - **Long iterative processes**: Small F32 errors compound; F64 maintains stability
///
/// # Resolution Order
///
/// `per-operation > tensor-level > client default`
///
/// # Default
///
/// F32 is the default (good balance for ML). Use F64 for scientific computing
/// where precision is critical. BF16 is experimental for research.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AccumulationPrecision {
    /// Accumulate in F64 (highest precision, for scientific computing)
    /// Essential for: physics simulations, CFD, climate modeling, orbital mechanics
    F64,
    /// Accumulate in F32 (default, good for ML/AI)
    /// Sufficient for: deep learning, general numeric computing
    #[default]
    F32,
    /// Accumulate in BF16 (experimental, for research)
    /// Use when: memory is extremely constrained, algorithm is numerically stable
    BF16,
}

// ============================================================================
// DType Enum
// ============================================================================

/// Data types supported by numr tensors
///
/// This enum represents the element type of a tensor at runtime.
/// Using an enum (rather than generics) allows:
/// - Mixed-precision operations
/// - Runtime type selection
/// - Support for quantized types that aren't `Copy`
///
/// # Discriminant Values (Serialization Stability)
///
/// The discriminant values are **stable** for serialization purposes:
/// - Floats: 0-9 (F64=0, F32=1, F16=2, BF16=3, FP8E4M3=4, FP8E5M2=5)
/// - Signed ints: 10-19 (I64=10, I32=11, I16=12, I8=13)
/// - Unsigned ints: 20-29 (U64=20, U32=21, U16=22, U8=23)
/// - Bool: 30
/// - Complex: 40-49 (Complex64=40, Complex128=41)
///
/// New types will use reserved ranges. Existing values are NEVER changed.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum DType {
    // Floating point types (0-9)
    /// 64-bit floating point
    F64 = 0,
    /// 32-bit floating point (most common)
    F32 = 1,
    /// 16-bit floating point (IEEE 754)
    F16 = 2,
    /// 16-bit brain floating point
    BF16 = 3,
    /// 8-bit floating point (1 sign + 4 exp + 3 mant), range ~[-448, 448]
    /// Best for: weights, activations (higher precision, smaller range)
    FP8E4M3 = 4,
    /// 8-bit floating point (1 sign + 5 exp + 2 mant), range ~[-57344, 57344]
    /// Best for: gradients (larger dynamic range, lower precision)
    FP8E5M2 = 5,

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

    // Complex types
    /// 64-bit complex (two f32: re, im)
    Complex64 = 40,
    /// 128-bit complex (two f64: re, im)
    Complex128 = 41,
}

impl DType {
    /// Size of one element in bytes
    #[inline]
    pub const fn size_in_bytes(self) -> usize {
        match self {
            Self::Complex128 => 16,
            Self::F64 | Self::I64 | Self::U64 | Self::Complex64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::FP8E4M3 | Self::FP8E5M2 | Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }

    /// Returns true if this is a floating point type
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::F64 | Self::F32 | Self::F16 | Self::BF16 | Self::FP8E4M3 | Self::FP8E5M2
        )
    }

    /// Returns true if this is a complex number type
    #[inline]
    pub const fn is_complex(self) -> bool {
        matches!(self, Self::Complex64 | Self::Complex128)
    }

    /// Returns the underlying float type for complex types
    /// Returns None for non-complex types
    #[inline]
    pub const fn complex_component_dtype(self) -> Option<Self> {
        match self {
            Self::Complex64 => Some(Self::F32),
            Self::Complex128 => Some(Self::F64),
            _ => None,
        }
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
        self.is_float() || self.is_signed_int() || self.is_complex()
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
            Self::FP8E4M3 => "fp8e4m3",
            Self::FP8E5M2 => "fp8e5m2",
            Self::I64 => "i64",
            Self::I32 => "i32",
            Self::I16 => "i16",
            Self::I8 => "i8",
            Self::U64 => "u64",
            Self::U32 => "u32",
            Self::U16 => "u16",
            Self::U8 => "u8",
            Self::Bool => "bool",
            Self::Complex64 => "c64",
            Self::Complex128 => "c128",
        }
    }

    /// Minimum value representable by this dtype (as f64)
    ///
    /// For complex types, returns the minimum value of each component
    pub fn min_value(self) -> f64 {
        match self {
            Self::F64 => f64::MIN,
            Self::F32 => f32::MIN as f64,
            Self::F16 => -65504.0,     // IEEE 754 half precision
            Self::BF16 => -3.4e38,     // Approximate
            Self::FP8E4M3 => -448.0,   // 1 sign + 4 exp + 3 mant
            Self::FP8E5M2 => -57344.0, // 1 sign + 5 exp + 2 mant
            Self::I64 => i64::MIN as f64,
            Self::I32 => i32::MIN as f64,
            Self::I16 => i16::MIN as f64,
            Self::I8 => i8::MIN as f64,
            Self::U64 => 0.0,
            Self::U32 => 0.0,
            Self::U16 => 0.0,
            Self::U8 => 0.0,
            Self::Bool => 0.0,
            // Complex types: component min
            Self::Complex64 => f32::MIN as f64,
            Self::Complex128 => f64::MIN,
        }
    }

    /// Maximum value representable by this dtype (as f64)
    ///
    /// For complex types, returns the maximum value of each component
    pub fn max_value(self) -> f64 {
        match self {
            Self::F64 => f64::MAX,
            Self::F32 => f32::MAX as f64,
            Self::F16 => 65504.0,
            Self::BF16 => 3.4e38,
            Self::FP8E4M3 => 448.0,
            Self::FP8E5M2 => 57344.0,
            Self::I64 => i64::MAX as f64,
            Self::I32 => i32::MAX as f64,
            Self::I16 => i16::MAX as f64,
            Self::I8 => i8::MAX as f64,
            Self::U64 => u64::MAX as f64,
            Self::U32 => u32::MAX as f64,
            Self::U16 => u16::MAX as f64,
            Self::U8 => u8::MAX as f64,
            Self::Bool => 1.0,
            // Complex types: component max
            Self::Complex64 => f32::MAX as f64,
            Self::Complex128 => f64::MAX,
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
            | (1 << DType::BF16 as u8)
            | (1 << DType::FP8E4M3 as u8)
            | (1 << DType::FP8E5M2 as u8),
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

    /// All complex types
    pub const COMPLEX: Self = Self {
        bits: (1 << DType::Complex64 as u8) | (1 << DType::Complex128 as u8),
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
        // FP8 types are 1 byte
        assert_eq!(DType::FP8E4M3.size_in_bytes(), 1);
        assert_eq!(DType::FP8E5M2.size_in_bytes(), 1);
    }

    #[test]
    fn test_dtype_categories() {
        assert!(DType::F32.is_float());
        assert!(!DType::I32.is_float());
        assert!(DType::I32.is_signed_int());
        assert!(DType::U32.is_unsigned_int());
        assert!(!DType::U32.is_signed());
        // FP8 types are floats
        assert!(DType::FP8E4M3.is_float());
        assert!(DType::FP8E5M2.is_float());
        assert!(DType::FP8E4M3.is_signed());
        assert!(DType::FP8E5M2.is_signed());
    }

    #[test]
    fn test_dtype_set() {
        assert!(DTypeSet::FLOATS.contains(DType::F32));
        assert!(!DTypeSet::FLOATS.contains(DType::I32));
        assert!(DTypeSet::INTS.contains(DType::I32));
        assert!(DTypeSet::NUMERIC.contains(DType::F32));
        assert!(DTypeSet::NUMERIC.contains(DType::I32));
        // FP8 types in FLOATS set
        assert!(DTypeSet::FLOATS.contains(DType::FP8E4M3));
        assert!(DTypeSet::FLOATS.contains(DType::FP8E5M2));
    }

    #[test]
    fn test_fp8_dtype_values() {
        // FP8E4M3: range ~[-448, 448]
        assert_eq!(DType::FP8E4M3.min_value(), -448.0);
        assert_eq!(DType::FP8E4M3.max_value(), 448.0);
        // FP8E5M2: range ~[-57344, 57344]
        assert_eq!(DType::FP8E5M2.min_value(), -57344.0);
        assert_eq!(DType::FP8E5M2.max_value(), 57344.0);
    }

    #[test]
    fn test_fp8_short_names() {
        assert_eq!(DType::FP8E4M3.short_name(), "fp8e4m3");
        assert_eq!(DType::FP8E5M2.short_name(), "fp8e5m2");
    }

    #[test]
    fn test_compute_precision_default() {
        assert_eq!(ComputePrecision::default(), ComputePrecision::BF16);
    }

    #[test]
    fn test_accumulation_precision_default() {
        assert_eq!(AccumulationPrecision::default(), AccumulationPrecision::F32);
    }
}
