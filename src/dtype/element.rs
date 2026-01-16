//! Element trait for mapping Rust types to DType

use super::DType;
use bytemuck::{Pod, Zeroable};

/// Trait for types that can be elements of a tensor
///
/// This trait connects Rust's type system to numr's runtime dtype system.
/// It's implemented for all primitive numeric types.
pub trait Element: Copy + Clone + Send + Sync + Pod + Zeroable + 'static {
    /// The corresponding DType for this Rust type
    const DTYPE: DType;

    /// Convert to f64 for generic numeric operations
    fn to_f64(self) -> f64;

    /// Convert from f64
    fn from_f64(v: f64) -> Self;

    /// Zero value
    fn zero() -> Self;

    /// One value
    fn one() -> Self;
}

impl Element for f64 {
    const DTYPE: DType = DType::F64;

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }

    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }

    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }
}

impl Element for i64 {
    const DTYPE: DType = DType::I64;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as i64
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for i32 {
    const DTYPE: DType = DType::I32;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as i32
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for i16 {
    const DTYPE: DType = DType::I16;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as i16
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for i8 {
    const DTYPE: DType = DType::I8;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as i8
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for u64 {
    const DTYPE: DType = DType::U64;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as u64
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for u32 {
    const DTYPE: DType = DType::U32;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as u32
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for u16 {
    const DTYPE: DType = DType::U16;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as u16
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for u8 {
    const DTYPE: DType = DType::U8;

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as u8
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

// Note: bool doesn't implement Pod, so we can't implement Element for it directly.
// Boolean tensors use u8 internally.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_dtype() {
        assert_eq!(f64::DTYPE, DType::F64);
        assert_eq!(f32::DTYPE, DType::F32);
        assert_eq!(i32::DTYPE, DType::I32);
        assert_eq!(u8::DTYPE, DType::U8);
    }

    #[test]
    fn test_element_conversions() {
        assert_eq!(f32::from_f64(3.14).to_f64(), 3.14f32 as f64);
        assert_eq!(i32::from_f64(42.0), 42);
    }
}
