//! Accumulation types of `resize` on the CPU: f32 and f64.
//!
//! Each method is one C++ expression of PyTorch's CPU kernels, with the
//! promotions that expression applies. `image_resize_taps` builds the windows
//! from them.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::dtype::Element;

/// Accumulation type of `resize`: f32 or f64.
pub trait ResizeAcc:
    Element
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Taps per block of rounded products in the antialias sum: the lanes of
    /// a 16-byte vector, as PyTorch's AVX2 build vectorizes that loop.
    const AA_BLOCK: usize;

    /// Exact constant. Every constant used here is a small dyadic value.
    fn constant(v: f64) -> Self;

    /// Integer to float, as C converts `int64_t`.
    fn from_i64(v: i64) -> Self;

    /// `a * b + c` with one rounding.
    fn fma(a: Self, b: Self, c: Self) -> Self;

    /// Absolute value.
    fn abs(self) -> Self;

    /// `(int64_t)floorf(x)`. PyTorch calls `floorf`, so F64 rounds to f32
    /// first.
    fn floor_index(self) -> i64;

    /// `(int)ceil(x)` for a positive `x`.
    fn ceil_count(self) -> usize;

    /// `scalar_t center = scale * (i + 0.5)`, the product taken in f64.
    fn aa_center(scale: Self, i: usize) -> Self;

    /// `(int64_t)(x + 0.5)`, the sum taken in f64.
    fn aa_bound(x: Self) -> i64;

    /// `(d + 0.5) * invscale`, both steps taken in f64.
    fn aa_arg(d: Self, invscale: Self) -> Self;

    /// Widen an element into the accumulation type.
    fn load<E: Element>(e: E) -> Self;

    /// Round into an element.
    fn store<E: Element>(self) -> E;
}

impl ResizeAcc for f32 {
    const AA_BLOCK: usize = 4;

    #[inline(always)]
    fn constant(v: f64) -> Self {
        v as f32
    }
    #[inline(always)]
    fn from_i64(v: i64) -> Self {
        v as f32
    }
    #[inline(always)]
    fn fma(a: Self, b: Self, c: Self) -> Self {
        a.mul_add(b, c)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f32::abs(self)
    }
    #[inline(always)]
    fn floor_index(self) -> i64 {
        self.floor() as i64
    }
    #[inline(always)]
    fn ceil_count(self) -> usize {
        self.ceil() as usize
    }
    #[inline(always)]
    fn aa_center(scale: Self, i: usize) -> Self {
        (f64::from(scale) * (i as f64 + 0.5)) as f32
    }
    #[inline(always)]
    fn aa_bound(x: Self) -> i64 {
        (f64::from(x) + 0.5) as i64
    }
    #[inline(always)]
    fn aa_arg(d: Self, invscale: Self) -> Self {
        ((f64::from(d) + 0.5) * f64::from(invscale)) as f32
    }
    #[inline(always)]
    fn load<E: Element>(e: E) -> Self {
        e.to_f32()
    }
    #[inline(always)]
    fn store<E: Element>(self) -> E {
        E::from_f32(self)
    }
}

impl ResizeAcc for f64 {
    const AA_BLOCK: usize = 2;

    #[inline(always)]
    fn constant(v: f64) -> Self {
        v
    }
    #[inline(always)]
    fn from_i64(v: i64) -> Self {
        v as f64
    }
    #[inline(always)]
    fn fma(a: Self, b: Self, c: Self) -> Self {
        a.mul_add(b, c)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline(always)]
    fn floor_index(self) -> i64 {
        (self as f32).floor() as i64
    }
    #[inline(always)]
    fn ceil_count(self) -> usize {
        self.ceil() as usize
    }
    #[inline(always)]
    fn aa_center(scale: Self, i: usize) -> Self {
        scale * (i as f64 + 0.5)
    }
    #[inline(always)]
    fn aa_bound(x: Self) -> i64 {
        (x + 0.5) as i64
    }
    #[inline(always)]
    fn aa_arg(d: Self, invscale: Self) -> Self {
        (d + 0.5) * invscale
    }
    #[inline(always)]
    fn load<E: Element>(e: E) -> Self {
        e.to_f64()
    }
    #[inline(always)]
    fn store<E: Element>(self) -> E {
        E::from_f64(self)
    }
}
