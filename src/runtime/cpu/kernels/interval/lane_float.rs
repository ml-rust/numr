//! Scalar float rules that match numr's CPU tensor primitives bit for bit.

use super::super::simd::next_representable::scalar::{step_u32, step_u64};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// F32 or F64 lane arithmetic.
///
/// Each method matches the CPU kernel of the tensor primitive it names.
pub trait LaneFloat:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Send
    + Sync
    + 'static
{
    /// Positive zero.
    const ZERO: Self;
    /// One.
    const ONE: Self;
    /// Smallest magnitude inside the guarded residual range.
    const GUARD_MIN: Self;
    /// Largest magnitude inside the guarded residual range.
    const GUARD_MAX: Self;
    /// Half the largest finite value.
    const HALF_MAX: Self;
    /// The largest finite value.
    const MAX: Self;

    /// Nearest value of `value`, exact for every probe constant.
    fn of(value: f64) -> Self;
    /// `next_up`.
    fn up(self) -> Self;
    /// `next_down`.
    fn down(self) -> Self;
    /// Neither NaN nor infinite.
    fn finite(self) -> bool;
    /// `abs`.
    fn magnitude(self) -> Self;
    /// `sqrt`.
    fn root(self) -> Self;
    /// `fused_mul_add(self, b, c)`.
    fn fused(self, b: Self, c: Self) -> Self;
    /// `trunc`.
    fn truncated(self) -> Self;
    /// The bit pattern of positive zero.
    fn positive_zero(self) -> bool;
}

impl LaneFloat for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const GUARD_MIN: Self = f32::from_bits((127 - 40) << 23);
    const GUARD_MAX: Self = f32::from_bits((127 + 40) << 23);
    const HALF_MAX: Self = f32::MAX * 0.5;
    const MAX: Self = f32::MAX;

    #[inline(always)]
    fn of(value: f64) -> Self {
        value as f32
    }
    #[inline(always)]
    fn up(self) -> Self {
        f32::from_bits(step_u32(self.to_bits(), true))
    }
    #[inline(always)]
    fn down(self) -> Self {
        f32::from_bits(step_u32(self.to_bits(), false))
    }
    #[inline(always)]
    fn finite(self) -> bool {
        self.is_finite()
    }
    #[inline(always)]
    fn magnitude(self) -> Self {
        self.abs()
    }
    #[inline(always)]
    fn root(self) -> Self {
        self.sqrt()
    }
    #[inline(always)]
    fn fused(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }
    #[inline(always)]
    fn truncated(self) -> Self {
        self.trunc()
    }
    #[inline(always)]
    fn positive_zero(self) -> bool {
        self.to_bits() == 0
    }
}

impl LaneFloat for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const GUARD_MIN: Self = f64::from_bits((1023 - 450) << 52);
    const GUARD_MAX: Self = f64::from_bits((1023 + 450) << 52);
    const HALF_MAX: Self = f64::MAX * 0.5;
    const MAX: Self = f64::MAX;

    #[inline(always)]
    fn of(value: f64) -> Self {
        value
    }
    #[inline(always)]
    fn up(self) -> Self {
        f64::from_bits(step_u64(self.to_bits(), true))
    }
    #[inline(always)]
    fn down(self) -> Self {
        f64::from_bits(step_u64(self.to_bits(), false))
    }
    #[inline(always)]
    fn finite(self) -> bool {
        self.is_finite()
    }
    #[inline(always)]
    fn magnitude(self) -> Self {
        self.abs()
    }
    #[inline(always)]
    fn root(self) -> Self {
        self.sqrt()
    }
    #[inline(always)]
    fn fused(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }
    #[inline(always)]
    fn truncated(self) -> Self {
        self.trunc()
    }
    #[inline(always)]
    fn positive_zero(self) -> bool {
        self.to_bits() == 0
    }
}

/// Strictly greater than zero. NaN is not positive.
#[inline(always)]
pub fn positive<T: LaneFloat>(value: T) -> bool {
    value > T::ZERO
}

/// `maximum`: the first operand only when it compares greater.
#[inline(always)]
pub fn max<T: LaneFloat>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

/// `minimum`: the first operand only when it compares less.
#[inline(always)]
pub fn min<T: LaneFloat>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

/// `where_cond`.
#[inline(always)]
pub fn select<T: LaneFloat>(condition: bool, x: T, y: T) -> T {
    if condition { x } else { y }
}

/// `two_sum`: rounded sum and represented residual.
#[inline(always)]
pub fn two_sum<T: LaneFloat>(a: T, b: T) -> (T, T) {
    let sum = a + b;
    let b_virtual = sum - a;
    let a_virtual = sum - b_virtual;
    let b_roundoff = b - b_virtual;
    let a_roundoff = a - a_virtual;
    (sum, a_roundoff + b_roundoff)
}

/// `midpoint`: the halved sum when it cannot overflow, else the sum of halves.
#[inline(always)]
pub fn midpoint<T: LaneFloat>(a: T, b: T) -> T {
    let half = T::of(0.5);
    let small = a.magnitude() <= T::HALF_MAX && b.magnitude() <= T::HALF_MAX;
    let sum = (select(small, a, T::ZERO) + select(small, b, T::ZERO)) * half;
    select(small, sum, a * half + b * half)
}

/// Magnitude inside the guarded residual range.
#[inline(always)]
pub fn guarded<T: LaneFloat>(value: T) -> bool {
    let magnitude = value.magnitude();
    magnitude >= T::GUARD_MIN && magnitude <= T::GUARD_MAX
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_constants_are_exact_powers_of_two() {
        assert_eq!(f32::GUARD_MIN as f64, 2f64.powi(-40));
        assert_eq!(f32::GUARD_MAX as f64, 2f64.powi(40));
        assert_eq!(f64::GUARD_MIN, 2f64.powi(-450));
        assert_eq!(f64::GUARD_MAX, 2f64.powi(450));
    }

    #[test]
    fn neighbors_match_integer_stepping() {
        assert_eq!(0.0f64.down().to_bits(), 0x8000_0000_0000_0001);
        assert_eq!((-0.0f32).up().to_bits(), 1);
        assert_eq!(f64::INFINITY.up(), f64::INFINITY);
        assert_eq!(f32::from_bits(0x7fc0_1234).up().to_bits(), 0x7fc0_1234);
    }
}
