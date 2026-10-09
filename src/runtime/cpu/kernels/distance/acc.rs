//! The accumulator a distance metric runs in.
//!
//! Every metric in this family builds a running total across the `d` components
//! of a pair of vectors. A total kept in the element type is wrong for every
//! float narrower than F32, for the reason
//! [`wide_acc`](crate::runtime::cpu::kernels::wide_acc) states: the accumulator
//! stops growing once its spacing exceeds twice the increment, so the terms at
//! the tail of a long vector are dropped entirely.
//!
//! [`DistAcc`] is the widening adapter that lets one metric body serve every
//! element type. It resolves to:
//!
//! - `f32` for F32 and for every `DType::is_narrow_float()` element (F16, BF16,
//!   FP8E4M3, FP8E5M2). This is the accumulator CUDA's `distance.cu` already
//!   uses — `AccType<__half>` and `AccType<__nv_bfloat16>` are both `float`.
//! - `f64` for F64, which f32 cannot hold.
//!
//! The accumulator width matches CUDA's. The summation order does not always
//! match. F32 and F64 inputs route through the SIMD kernels in
//! [`simd::distance`](crate::runtime::cpu::kernels::simd::distance), which sum
//! in several lanes and fuse multiply-adds. Their results can differ from
//! CUDA's and from the sequential loop in the last bits. The narrow floats keep
//! the sequential loop.
//!
//! Widening and narrowing for the f32 accumulator go through
//! [`WideAcc`](crate::runtime::cpu::kernels::wide_acc::WideAcc) rather than
//! open-coding a second conversion; the f64 accumulator has no `WideAcc` impl
//! (nothing in numr is wider than F64) and uses `Element`'s own f64 conversion.
//!
//! Output dtypes never change: the accumulator narrows back to `T` exactly once,
//! at write-out.

use super::super::wide_acc::WideAcc;
use crate::dtype::Element;
use num_traits::Float;

/// An accumulator wide enough to hold a distance metric's running total for
/// element type `T`.
///
/// `'static` lets `metrics` test the accumulator's exact type with `TypeId`
/// before it routes to an f32 or f64 SIMD kernel.
pub trait DistAcc<T: Element>: Float + 'static {
    /// Widen one element into the accumulator.
    fn widen(v: T) -> Self;

    /// Narrow the accumulator back to the element type, for write-out.
    fn narrow(self) -> T;

    /// The Minkowski exponent in the accumulator's precision.
    ///
    /// Taking `p` straight from f64 keeps it unrounded for the narrow floats,
    /// which is what CUDA does — its kernels take `p` as a `float` argument
    /// regardless of the element type.
    fn exponent(p: f64) -> Self;

    /// A component count (the vector length `d`) in the accumulator's precision.
    fn count(n: usize) -> Self;

    /// An f64 value in the accumulator's precision.
    ///
    /// The SIMD kernels return f32 or f64 sums. An f32 sum widened to f64 and
    /// passed here comes back unchanged.
    fn from_f64(v: f64) -> Self;
}

impl<T: Element> DistAcc<T> for f32 {
    #[inline]
    fn widen(v: T) -> Self {
        <f32 as WideAcc>::from_elem(v)
    }

    #[inline]
    fn narrow(self) -> T {
        <f32 as WideAcc>::to_elem::<T>(self)
    }

    #[inline]
    fn exponent(p: f64) -> Self {
        p as f32
    }

    #[inline]
    fn count(n: usize) -> Self {
        n as f32
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl<T: Element> DistAcc<T> for f64 {
    #[inline]
    fn widen(v: T) -> Self {
        v.to_f64()
    }

    #[inline]
    fn narrow(self) -> T {
        T::from_f64(self)
    }

    #[inline]
    fn exponent(p: f64) -> Self {
        p
    }

    #[inline]
    fn count(n: usize) -> Self {
        n as f64
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "f16")]
    #[test]
    fn f32_accumulator_keeps_terms_an_f16_accumulator_drops() {
        // f16 steps by 1.0 across [1024, 2048), so 1024 + 0.25 rounds back to
        // 1024 and every later term is lost. The f32 accumulator keeps them.
        let big = half::f16::from_f32(1024.0);
        let small = half::f16::from_f32(0.25);

        let mut wide: f32 = <f32 as DistAcc<half::f16>>::widen(big);
        let mut narrow = big;
        for _ in 0..64 {
            wide += <f32 as DistAcc<half::f16>>::widen(small);
            narrow += small;
        }

        assert_eq!(wide, 1040.0);
        assert_eq!(narrow.to_f32(), 1024.0);
        assert_eq!(
            <f32 as DistAcc<half::f16>>::narrow(wide).to_f32(),
            1040.0,
            "1040 is exactly representable in f16, so only the accumulator width shows here"
        );
    }

    #[test]
    fn f64_accumulator_is_exact_for_f64_elements() {
        let v = 1.0e300f64;
        assert_eq!(<f64 as DistAcc<f64>>::widen(v), v);
        assert_eq!(<f64 as DistAcc<f64>>::narrow(v), v);
    }
}
