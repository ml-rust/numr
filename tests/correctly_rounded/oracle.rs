//! MPFR reference for the correctly rounded binary64 functions.

use numr::algorithm::correctly_rounded::{self as cr, CANONICAL_NAN};
use rug::Float;
use rug::float::Round;

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
