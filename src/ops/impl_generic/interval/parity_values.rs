//! Special values and bit comparisons for interval parity tests.

use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Float element with special test values and raw bits.
pub trait TestFloat: Element + bytemuck::Pod + Copy + std::fmt::Debug {
    /// NaNs, signed zeros, infinities, subnormals, extremes, and ordinary values.
    fn specials() -> Vec<Self>;
    /// Raw encoding widened to u64.
    fn bits(self) -> u64;
}

impl TestFloat for f64 {
    fn specials() -> Vec<Self> {
        vec![
            -0.0,
            0.0,
            1.0,
            -1.0,
            0.1,
            -2.5,
            3.0,
            1.0 + f64::EPSILON,
            1e300,
            -1e-300,
            2f64.powi(-460),
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            -f64::from_bits(3),
            f64::MAX,
            -f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from_bits(0xfff4_0000_0000_1234),
        ]
    }

    fn bits(self) -> u64 {
        self.to_bits()
    }
}

impl TestFloat for f32 {
    fn specials() -> Vec<Self> {
        vec![
            -0.0,
            0.0,
            1.0,
            -1.0,
            0.1,
            -2.5,
            3.0,
            1.0 + f32::EPSILON,
            1e30,
            -1e-30,
            2f32.powi(-45),
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(3),
            f32::MAX,
            -f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0xffa0_1234),
        ]
    }

    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

/// Deterministic index stream.
pub struct Lcg(pub u64);

impl Lcg {
    /// Next index below `bound`.
    pub fn below(&mut self, bound: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as usize) % bound
    }
}

/// Raw element bits of a tensor.
pub fn bits<R: Runtime<DType = DType>, T: TestFloat>(tensor: &Tensor<R>) -> Result<Vec<u64>> {
    Ok(tensor.try_to_vec::<T>()?.into_iter().map(T::bits).collect())
}

/// Status flag value.
pub fn flag<R: Runtime<DType = DType>>(status: &Tensor<R>) -> Result<u32> {
    status.item::<u32>()
}

/// Assert identical shapes and element bits.
pub fn assert_same<R: Runtime<DType = DType>, T: TestFloat>(
    native: &Tensor<R>,
    reference: &Tensor<R>,
    label: &str,
) -> Result<()> {
    if R::name() == "cuda" {
        // GPU compilers commute operands, so which of several NaN inputs propagates is not fixed.
        return assert_same_nan_class::<R, T>(native, reference, label);
    }
    assert_eq!(native.shape(), reference.shape(), "{label} shape");
    let (native_bits, reference_bits) = (bits::<R, T>(native)?, bits::<R, T>(reference)?);
    for (lane, (n, r)) in native_bits.iter().zip(&reference_bits).enumerate() {
        assert_eq!(n, r, "{label} lane {lane}: native {n:#x} reference {r:#x}");
    }
    Ok(())
}

/// Assert identical bits, except that any two NaNs compare equal.
///
/// Fused multiply-add picks among several NaN operands by instruction form,
/// so SIMD and scalar paths of the same composition can return different NaNs.
pub fn assert_same_nan_class<R: Runtime<DType = DType>, T: TestFloat>(
    native: &Tensor<R>,
    reference: &Tensor<R>,
    label: &str,
) -> Result<()> {
    assert_eq!(native.shape(), reference.shape(), "{label} shape");
    let (n_values, r_values) = (native.try_to_vec::<T>()?, reference.try_to_vec::<T>()?);
    for (lane, (n, r)) in n_values.iter().zip(&r_values).enumerate() {
        let both_nan = n.to_f64().is_nan() && r.to_f64().is_nan();
        assert!(
            both_nan || n.bits() == r.bits(),
            "{label} lane {lane}: native {:#x} reference {:#x}",
            n.bits(),
            r.bits()
        );
    }
    Ok(())
}
