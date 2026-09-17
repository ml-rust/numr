//! Accumulator trait for precision-aware reductions.
//!
//! This allows a single generic implementation for both FP32 and FP64 accumulation,
//! avoiding code duplication while maintaining type safety and performance.
//!
//! Uses `Into<f64>` for output conversion, `acc_in` for input (f64 -> Self).

/// Trait for accumulation types (f32, f64) used in precision-aware reductions.
pub trait Accumulator: Copy + PartialOrd + PartialEq + Into<f64> {
    const ZERO: Self;
    const ONE: Self;
    /// Convert f64 input to accumulator type
    fn acc_in(v: f64) -> Self;
    fn acc_add(self, other: Self) -> Self;
    fn acc_mul(self, other: Self) -> Self;
    fn acc_div(self, n: usize) -> Self;
}

impl Accumulator for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn acc_in(v: f64) -> Self {
        v as f32
    }
    #[inline]
    fn acc_add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn acc_mul(self, other: Self) -> Self {
        self * other
    }
    #[inline]
    fn acc_div(self, n: usize) -> Self {
        self / n as f32
    }
}

impl Accumulator for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn acc_in(v: f64) -> Self {
        v
    }
    #[inline]
    fn acc_add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn acc_mul(self, other: Self) -> Self {
        self * other
    }
    #[inline]
    fn acc_div(self, n: usize) -> Self {
        self / n as f64
    }
}
