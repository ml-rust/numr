//! Shared checks for the distance dispatch tests.
//!
//! Every check runs each level this CPU supports and compares it to a
//! reference. The error bound is derived, not tuned. See [`bound`].

use crate::runtime::cpu::kernels::simd::{SimdLevel, detect_simd};

/// A float width under test.
pub trait TestFloat: Copy + std::fmt::Debug {
    /// Unit roundoff: `2^-24` for f32, `2^-53` for f64.
    const EPS: f64;
    /// Smallest positive subnormal: `2^-149` for f32, `2^-1074` for f64.
    const ETA: f64;
    /// Lanes in one AVX2 register.
    const LANE: usize;
    /// Moves values in `[-1, 1]` into the subnormal range.
    const SUBNORMAL_SCALE: f64;
    /// Rounds an f64 into this width.
    fn from_f64(v: f64) -> Self;
    /// Widens this value into f64, exactly.
    fn to_f64(self) -> f64;
}

impl TestFloat for f32 {
    const EPS: f64 = f32::EPSILON as f64 / 2.0;
    const ETA: f64 = f32::from_bits(1) as f64;
    const LANE: usize = 8;
    const SUBNORMAL_SCALE: f64 = f32::MIN_POSITIVE as f64 / 16.0;
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl TestFloat for f64 {
    const EPS: f64 = f64::EPSILON / 2.0;
    const ETA: f64 = f64::from_bits(1);
    const LANE: usize = 4;
    const SUBNORMAL_SCALE: f64 = f64::MIN_POSITIVE / 16.0;
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// A kernel under test, called at an explicit level.
pub type Kernel<F> = fn(SimdLevel, *const F, *const F, usize) -> F;

/// One summed term, computed in f64 from the widened inputs.
pub type Term = fn(f64, f64) -> f64;

/// Which input receives the injected NaN or infinity.
#[derive(Copy, Clone, Debug)]
pub enum Side {
    /// Inject into `a`.
    A,
    /// Inject into `b`.
    B,
}

/// Every level this CPU can run, `Scalar` first.
///
/// `detect_simd()` is the upper bound. AVX2+FMA joins only when the CPU
/// reports both features, so no test runs an unsupported instruction. On
/// AArch64 the detected NEON level joins, so the tests run the NEON kernels.
pub fn levels() -> Vec<SimdLevel> {
    let top = detect_simd();
    let mut levels = vec![SimdLevel::Scalar];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            levels.push(SimdLevel::Avx2Fma);
            if top == SimdLevel::Avx512 {
                levels.push(top);
            }
        }
    }
    if top.is_arm64() {
        levels.push(top);
    }
    levels
}

/// Lengths around each loop boundary of the AVX2 and NEON kernels, plus two
/// long ones. The result is sorted and has no duplicates.
///
/// `lane` is the AVX2 lane count. For AVX2, `3 * lane` is the cosine main-loop
/// step and `4 * lane` is the step of the others. A NEON register holds
/// `lane / 2` elements, and every NEON main loop steps `2 * lane`.
pub fn lengths(lane: usize) -> Vec<usize> {
    let neon = lane / 2;
    let mut lengths = vec![
        0,
        1,
        neon - 1,
        neon,
        neon + 1,
        lane - 1,
        lane,
        lane + 1,
        2 * lane - 1,
        2 * lane,
        2 * lane + 1,
        2 * lane + 3,
        3 * lane - 1,
        3 * lane,
        3 * lane + 1,
        4 * lane - 1,
        4 * lane,
        4 * lane + 1,
        1536,
        1537,
    ];
    lengths.sort_unstable();
    lengths.dedup();
    lengths
}

/// A 64-bit linear congruential generator (Knuth's MMIX constants).
pub struct Lcg(u64);

impl Lcg {
    /// A generator seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next value, uniform in `[-1, 1)`.
    pub fn next_unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

/// `n` values uniform in `[-scale, scale)`, rounded into `F`.
pub fn random_vec<F: TestFloat>(rng: &mut Lcg, n: usize, scale: f64) -> Vec<F> {
    (0..n)
        .map(|_| F::from_f64(rng.next_unit() * scale))
        .collect()
}

/// Neumaier-compensated sum of `term(a[i], b[i])`, and `sum(|term|)`.
///
/// The terms run in f64. Compensated summation keeps the reference error near
/// `2 * 2^-53 * sum(|term|)`. That is far below the f32 bound. For f64 inputs
/// it adds 3 units of roundoff at most, which [`bound`] covers.
pub fn reference<F: TestFloat>(a: &[F], b: &[F], term: Term) -> (f64, f64) {
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
/// Recursive summation in any order has error at most `(n - 1) * eps * sum|t|`.
/// Rounding each term adds at most `3 * eps * |t|`: one difference and one
/// product, squared for sqeuclidean. The reference adds `2 * eps * sum|t|`.
/// `(len + 4) * eps * sum|t|` covers all of these.
///
/// Gradual underflow loses at most `eta / 2` per operation, and a kernel runs
/// fewer than `len + 4` operations per term chain. `(len + 4) * eta` covers it.
pub fn bound<F: TestFloat>(len: usize, abs_sum: f64) -> f64 {
    let n = len as f64 + 4.0;
    n * F::EPS * abs_sum + n * F::ETA
}

/// Every level matches [`reference`] within [`bound`], at both offsets.
///
/// Offset 1 shifts both slices one element into a larger buffer. That makes
/// every vector load unaligned.
pub fn check_reference<F: TestFloat>(op: &str, kernel: Kernel<F>, term: Term, scale: f64) {
    let mut rng = Lcg::new(0x9E37_79B9_7F4A_7C15);
    for len in lengths(F::LANE) {
        for offset in [0usize, 1] {
            let a_buf = random_vec::<F>(&mut rng, len + offset, scale);
            let b_buf = random_vec::<F>(&mut rng, len + offset, scale);
            let (a, b) = (&a_buf[offset..], &b_buf[offset..]);
            let (want, abs_sum) = reference(a, b, term);
            let tol = bound::<F>(len, abs_sum);
            for level in levels() {
                let got = kernel(level, a.as_ptr(), b.as_ptr(), len).to_f64();
                assert!(
                    got.is_finite() && (got - want).abs() <= tol,
                    "{op}: level {level}, len {len}, offset {offset}: \
                     got {got:e}, reference {want:e}, bound {tol:e}"
                );
            }
        }
    }
}

/// Every level returns exactly zero for two all-zero vectors.
pub fn check_zero<F: TestFloat>(op: &str, kernel: Kernel<F>) {
    for len in lengths(F::LANE) {
        let zeros = vec![F::from_f64(0.0); len];
        for level in levels() {
            let got = kernel(level, zeros.as_ptr(), zeros.as_ptr(), len).to_f64();
            assert!(got == 0.0, "{op}: level {level}, len {len}: got {got:e}");
        }
    }
}

/// A length whose AVX2 run enters the main loop, the one-register loop and the
/// scalar tail, with the positions that land in each.
fn special_layout(lane: usize) -> (usize, [usize; 4]) {
    let len = 5 * lane + 3;
    (len, [0, lane + 1, 4 * lane + 1, len - 1])
}

/// Runs `kernel` with `value` injected at `pos` on `side`.
fn run_injected<F: TestFloat>(
    kernel: Kernel<F>,
    level: SimdLevel,
    len: usize,
    pos: usize,
    side: Side,
    value: f64,
) -> f64 {
    let mut rng = Lcg::new(0x2545_F491_4F6C_DD1D);
    let mut a = random_vec::<F>(&mut rng, len, 1.0);
    let mut b = random_vec::<F>(&mut rng, len, 1.0);
    match side {
        Side::A => a[pos] = F::from_f64(value),
        Side::B => b[pos] = F::from_f64(value),
    }
    kernel(level, a.as_ptr(), b.as_ptr(), len).to_f64()
}

/// A NaN at any position on `side` makes the result NaN at every level.
pub fn check_nan<F: TestFloat>(op: &str, kernel: Kernel<F>, side: Side) {
    let (len, positions) = special_layout(F::LANE);
    for pos in positions {
        for level in levels() {
            let got = run_injected(kernel, level, len, pos, side, f64::NAN);
            assert!(
                got.is_nan(),
                "{op}: level {level}, len {len}, NaN at {pos} in {side:?}: got {got:e}"
            );
        }
    }
}

/// A `+inf` on `side` gives the same class (infinite or NaN) at every level as
/// at `Scalar`. An infinite result also keeps its sign.
pub fn check_inf<F: TestFloat>(op: &str, kernel: Kernel<F>, side: Side) {
    let (len, positions) = special_layout(F::LANE);
    for pos in positions {
        let want = run_injected(kernel, SimdLevel::Scalar, len, pos, side, f64::INFINITY);
        assert!(!want.is_finite(), "{op}: scalar stayed finite: {want:e}");
        for level in levels() {
            let got = run_injected(kernel, level, len, pos, side, f64::INFINITY);
            let same = (got.is_nan() && want.is_nan()) || got == want;
            assert!(
                same,
                "{op}: level {level}, len {len}, +inf at {pos} in {side:?}: \
                 got {got:e}, scalar {want:e}"
            );
        }
    }
}

/// The full check set for one kernel and one width.
///
/// `sides` lists the inputs whose NaN or infinity reaches this sum.
pub fn check_all<F: TestFloat>(op: &str, kernel: Kernel<F>, term: Term, sides: &[Side]) {
    check_reference(op, kernel, term, 1.0);
    check_reference(op, kernel, term, F::SUBNORMAL_SCALE);
    check_zero(op, kernel);
    for &side in sides {
        check_nan(op, kernel, side);
        check_inf(op, kernel, side);
    }
}
