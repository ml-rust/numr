//! Shared fixtures for the `numr::distance` slice API tests.

use numr::distance::{Kernels, SimdLevel};

pub(crate) const ALL_LEVELS: [SimdLevel; 5] = [
    SimdLevel::Scalar,
    SimdLevel::Avx2Fma,
    SimdLevel::Avx512,
    SimdLevel::Neon,
    SimdLevel::NeonFp16,
];

pub(crate) const LENGTHS: [usize; 20] = [
    0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 1535, 1536, 1537,
];

/// Every handle this CPU can build, `Scalar` first.
pub(crate) fn kernels() -> Vec<Kernels> {
    ALL_LEVELS
        .iter()
        .filter_map(|&level| Kernels::with_level(level))
        .collect()
}

/// A 64-bit linear congruential generator (Knuth's MMIX constants).
pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Uniform in `[-1, 1)`.
    pub(crate) fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    pub(crate) fn next_i8(&mut self) -> i8 {
        (self.next_u64() >> 56) as u8 as i8
    }
}

/// A float width under test.
pub(crate) trait TestFloat: Copy + std::fmt::Debug {
    /// Unit roundoff.
    const EPS: f64;
    /// Smallest positive subnormal.
    const ETA: f64;
    /// Moves values in `[-1, 1]` into the subnormal range.
    const SUBNORMAL_SCALE: f64;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn bits(self) -> u64;
}

impl TestFloat for f32 {
    const EPS: f64 = f32::EPSILON as f64 / 2.0;
    const ETA: f64 = f32::from_bits(1) as f64;
    const SUBNORMAL_SCALE: f64 = f32::MIN_POSITIVE as f64 / 16.0;
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn bits(self) -> u64 {
        self.to_bits() as u64
    }
}

impl TestFloat for f64 {
    const EPS: f64 = f64::EPSILON / 2.0;
    const ETA: f64 = f64::from_bits(1);
    const SUBNORMAL_SCALE: f64 = f64::MIN_POSITIVE / 16.0;
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

pub(crate) fn random_vec<F: TestFloat>(rng: &mut Lcg, n: usize, scale: f64) -> Vec<F> {
    (0..n)
        .map(|_| F::from_f64(rng.next_unit() * scale))
        .collect()
}

/// Neumaier-compensated sum of `term(a[i], b[i])` in f64, and `sum(|term|)`.
pub(crate) fn reference<F: TestFloat>(a: &[F], b: &[F], term: fn(f64, f64) -> f64) -> (f64, f64) {
    let (mut sum, mut comp, mut abs_sum) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let t = term(x.to_f64(), y.to_f64());
        abs_sum += t.abs();
        let s = sum + t;
        if sum.abs() >= t.abs() {
            comp += (sum - s) + t;
        } else {
            comp += (t - s) + sum;
        }
        sum = s;
    }
    (sum + comp, abs_sum)
}

/// Largest error a correct kernel can show against [`reference`].
///
/// Summation in any order: `(n - 1) * eps * sum|t|`. Rounding each term: at
/// most `3 * eps * |t|`. The reference: `2 * eps * sum|t|`. Gradual underflow:
/// `eta / 2` per operation, fewer than `len + 4` per term chain.
pub(crate) fn bound<F: TestFloat>(len: usize, abs_sum: f64) -> f64 {
    let n = len as f64 + 4.0;
    n * F::EPS * abs_sum + n * F::ETA
}

pub(crate) fn dot_term(a: f64, b: f64) -> f64 {
    a * b
}

pub(crate) fn l2_term(a: f64, b: f64) -> f64 {
    (a - b) * (a - b)
}

pub(crate) fn manhattan_term(a: f64, b: f64) -> f64 {
    (a - b).abs()
}
