//! `exp` range thresholds computed from MPFR, checked against the kernel's bit constants.

use super::oracle::Function;

/// Smallest `x` with `exp(x) = +inf`.
const OVERFLOW_FIRST: u64 = 0x4086_2e42_fefa_39f0;
/// Largest `x` with `exp(x) = +0`: `-0x1.74910d52d3052p+9`.
const UNDERFLOW_LAST: u64 = 0xc087_4910_d52d_3052;
/// Smallest `x` with `exp(x) >= 2^-1022`: `-0x1.6232bdd7abcd2p+9`.
const NORMAL_FIRST: u64 = 0xc086_232b_dd7a_bcd2;

/// Smallest bit pattern in `lo..=hi` for which `pred` holds. `pred` is monotone on the range.
fn first_true(mut lo: u64, mut hi: u64, pred: impl Fn(u64) -> bool) -> u64 {
    assert!(pred(hi) && !pred(lo));
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if pred(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

fn mpfr_exp(bits: u64) -> f64 {
    Function::Exp.reference(f64::from_bits(bits))
}

#[test]
fn test_exp_overflow_threshold_matches_mpfr() {
    let first = first_true(700f64.to_bits(), 720f64.to_bits(), |b| {
        mpfr_exp(b).is_infinite()
    });
    assert_eq!(
        first,
        OVERFLOW_FIRST,
        "first overflowing input {:e}",
        f64::from_bits(first)
    );
    assert_eq!(Function::Exp.numr(f64::from_bits(first)), f64::INFINITY);
    assert!(Function::Exp.numr(f64::from_bits(first - 1)).is_finite());
}

#[test]
fn test_exp_underflow_threshold_matches_mpfr() {
    // Negative bit patterns grow with magnitude, so exp(x) == 0 holds from the threshold up.
    let last = first_true((-740f64).to_bits(), (-750f64).to_bits(), |b| {
        mpfr_exp(b) == 0.0
    });
    assert_eq!(
        last,
        UNDERFLOW_LAST,
        "last zero input {:e}",
        f64::from_bits(last)
    );
    assert_eq!(Function::Exp.numr(f64::from_bits(last)).to_bits(), 0);
    assert!(Function::Exp.numr(f64::from_bits(last - 1)) > 0.0);
}

#[test]
fn test_exp_subnormal_threshold_matches_mpfr() {
    let first_subnormal = first_true((-700f64).to_bits(), (-720f64).to_bits(), |b| {
        mpfr_exp(b) < f64::MIN_POSITIVE
    });
    assert_eq!(first_subnormal - 1, NORMAL_FIRST);
    assert!(Function::Exp.numr(f64::from_bits(NORMAL_FIRST)) >= f64::MIN_POSITIVE);
    assert!(Function::Exp.numr(f64::from_bits(first_subnormal)) < f64::MIN_POSITIVE);
}
