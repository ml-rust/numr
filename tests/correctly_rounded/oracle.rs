//! MPFR reference for the correctly rounded binary64 functions.

use numr::algorithm::correctly_rounded::{self as cr, CANONICAL_NAN};
use rug::Float;
use rug::float::Round;
use rug::ops::PowAssignRound;

/// A function under test.
#[derive(Debug, Clone, Copy)]
pub enum Function {
    Exp,
    Log,
    Log1p,
}

impl Function {
    pub const ALL: [Function; 3] = [Function::Exp, Function::Log, Function::Log1p];

    /// The numr host function.
    pub fn numr(self, x: f64) -> f64 {
        match self {
            Function::Exp => cr::exp(x),
            Function::Log => cr::log(x),
            Function::Log1p => cr::log1p(x),
        }
    }

    /// The correctly rounded binary64 result from MPFR, with NaN canonical.
    pub fn reference(self, x: f64) -> f64 {
        if x.is_nan() {
            return CANONICAL_NAN;
        }
        let mut y = Float::with_val(53, x);
        let ternary = match self {
            Function::Exp => y.exp_round(Round::Nearest),
            Function::Log => y.ln_round(Round::Nearest),
            Function::Log1p => y.ln_1p_round(Round::Nearest),
        };
        y.subnormalize_ieee_round(ternary, Round::Nearest);
        let value = y.to_f64();
        if value.is_nan() { CANONICAL_NAN } else { value }
    }
}

/// Compare every input against MPFR bit for bit and panic with the first mismatches.
pub fn check_all<I: IntoIterator<Item = f64>>(function: Function, inputs: I, label: &str) {
    let mut checked = 0usize;
    let mut mismatches = Vec::new();
    for x in inputs {
        checked += 1;
        let got = function.numr(x);
        let want = function.reference(x);
        if got.to_bits() != want.to_bits() {
            mismatches.push((x, got, want));
        }
    }
    assert!(checked > 0, "{label}: no inputs");
    if !mismatches.is_empty() {
        let shown: Vec<String> = mismatches
            .iter()
            .take(10)
            .map(|(x, got, want)| {
                format!(
                    "x={x:e} ({:#018x}) numr={:#018x} mpfr={:#018x}",
                    x.to_bits(),
                    got.to_bits(),
                    want.to_bits()
                )
            })
            .collect();
        panic!(
            "{function:?} {label}: {} of {checked} inputs differ from MPFR. Fix the kernel; first cases:\n{}",
            mismatches.len(),
            shown.join("\n")
        );
    }
}

/// MPFR's correctly rounded binary64 `x^y`, with NaN canonical.
///
/// MPFR decides every special case: it follows C Annex F, so `pow(NaN, 0) = 1` and
/// `pow(1, NaN) = 1`. It has no signalling NaN, so those cases give 1 for every NaN.
pub fn pow_reference(x: f64, y: f64) -> f64 {
    let mut z = Float::with_val(53, x);
    let ternary = z.pow_assign_round(&Float::with_val(53, y), Round::Nearest);
    z.subnormalize_ieee_round(ternary, Round::Nearest);
    let value = z.to_f64();
    if value.is_nan() { CANONICAL_NAN } else { value }
}

/// Compare `pow` on every pair against MPFR bit for bit and panic with the first mismatches.
pub fn check_pow<I: IntoIterator<Item = (f64, f64)>>(pairs: I, label: &str) {
    let mut checked = 0usize;
    let mut mismatches = Vec::new();
    for (x, y) in pairs {
        checked += 1;
        let got = cr::pow(x, y);
        let want = pow_reference(x, y);
        if got.to_bits() != want.to_bits() {
            mismatches.push((x, y, got, want));
        }
    }
    assert!(checked > 0, "{label}: no inputs");
    if !mismatches.is_empty() {
        let shown: Vec<String> = mismatches
            .iter()
            .take(10)
            .map(|(x, y, got, want)| {
                format!(
                    "x={x:e} ({:#018x}) y={y:e} ({:#018x}) numr={:#018x} mpfr={:#018x}",
                    x.to_bits(),
                    y.to_bits(),
                    got.to_bits(),
                    want.to_bits()
                )
            })
            .collect();
        panic!(
            "pow {label}: {} of {checked} pairs differ from MPFR. Fix the kernel; first cases:\n{}",
            mismatches.len(),
            shown.join("\n")
        );
    }
}

#[test]
fn test_pow_oracle_returns_known_correctly_rounded_values() {
    assert_eq!(pow_reference(2.0, 0.5), std::f64::consts::SQRT_2);
    assert_eq!(pow_reference(f64::NAN, 0.0), 1.0);
    assert_eq!(pow_reference(1.0, f64::NAN), 1.0);
    assert_eq!(
        pow_reference(-8.0, 1.0 / 3.0).to_bits(),
        CANONICAL_NAN.to_bits()
    );
    assert_eq!(pow_reference(-0.0, -1.0), f64::NEG_INFINITY);
    assert_eq!(pow_reference(-2.0, 3.0), -8.0);
    assert_eq!(pow_reference(2.0, -1074.0).to_bits(), 1);
    assert_eq!(pow_reference(2.0, -1075.0).to_bits(), 0);
    assert_eq!(pow_reference(-2.0, -1075.0).to_bits(), 1 << 63);
    assert_eq!(pow_reference(10.0, 400.0), f64::INFINITY);
    // 3^34 is a 54-bit midpoint, rounded to even.
    assert_eq!(pow_reference(3.0, 34.0), 16_677_181_699_666_568.0);
}

#[test]
fn test_oracle_returns_known_correctly_rounded_values() {
    use std::f64::consts::{E, LN_2, LN_10};
    assert_eq!(Function::Exp.reference(1.0), E);
    assert_eq!(Function::Exp.reference(-0.0), 1.0);
    assert_eq!(Function::Exp.reference(710.0), f64::INFINITY);
    assert_eq!(Function::Exp.reference(-746.0).to_bits(), 0);
    // exp(-745) = 2^-1074.8..., which rounds to the smallest subnormal.
    assert_eq!(Function::Exp.reference(-745.0).to_bits(), 1);
    assert_eq!(Function::Log.reference(10.0), LN_10);
    assert_eq!(Function::Log.reference(-0.0), f64::NEG_INFINITY);
    assert_eq!(
        Function::Log.reference(-1.0).to_bits(),
        CANONICAL_NAN.to_bits()
    );
    assert_eq!(Function::Log1p.reference(1.0), LN_2);
    assert_eq!(
        Function::Log1p.reference(-0.0).to_bits(),
        (-0.0f64).to_bits()
    );
}

/// SplitMix64: a fixed-seed generator for reproducible samples.
pub struct SplitMix(pub u64);

impl SplitMix {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}
