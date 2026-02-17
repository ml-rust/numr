//! Efficient bitset for DType membership testing.

use super::DType;

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
