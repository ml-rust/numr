//! `pow` overflow, underflow and subnormal-result boundaries, against MPFR.
//!
//! The boundaries depend on the base, so each is found by bisection on MPFR's results.

use super::oracle::{SplitMix, check_pow, pow_reference};

/// Smallest bit pattern in `lo..=hi` for which `pred` holds. `pred` is monotone on the range.
fn first_true(mut lo: u64, mut hi: u64, pred: impl Fn(u64) -> bool) -> u64 {
    assert!(pred(hi) && !pred(lo), "{lo:#x}..{hi:#x} does not bracket");
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

/// `bits` stepped up to `reach` units in the last place, both directions.
fn around(bits: u64, reach: u64) -> impl Iterator<Item = f64> {
    (bits.saturating_sub(reach)..=bits + reach).map(f64::from_bits)
}

const BASES: [f64; 10] = [
    2.0,
    10.0,
    std::f64::consts::E,
    1.5,
    3.0,
    1.000_001,
    1.0 + f64::EPSILON,
    1.0e100,
    f64::MAX,
    1.25e-5,
];

/// For base `x`, the first exponents whose result is `inf`, subnormal and `0`.
fn thresholds(x: f64) -> [u64; 3] {
    let ax = x.abs();
    let (up, down) = if ax > 1.0 { (1.0, -1.0) } else { (-1.0, 1.0) };
    // Positive (or negative) y bit patterns grow with |y|, so each predicate is monotone.
    let span = |sign: f64| ((1.0e-300 * sign).to_bits(), (1.0e300 * sign).to_bits());
    let (lo, hi) = span(up);
    let overflow = first_true(lo, hi, |b| {
        pow_reference(ax, f64::from_bits(b)).is_infinite()
    });
    let (lo, hi) = span(down);
    let subnormal = first_true(lo, hi, |b| {
        pow_reference(ax, f64::from_bits(b)) < f64::MIN_POSITIVE
    });
    let zero = first_true(lo, hi, |b| pow_reference(ax, f64::from_bits(b)) == 0.0);
    [overflow, subnormal, zero]
}

#[test]
fn test_pow_overflow_and_underflow_thresholds_match_mpfr() {
    let mut pairs = Vec::new();
    for x in BASES {
        for threshold in thresholds(x) {
            for y in around(threshold, 64) {
                pairs.push((x, y));
                pairs.push((1.0 / x, -y));
            }
        }
    }
    check_pow(pairs, "range thresholds");
}

#[test]
fn test_pow_negative_base_thresholds_keep_the_sign() {
    // Odd integer exponents near the overflow and underflow edges of -2 and -3.
    let mut pairs = Vec::new();
    for x in [-2.0f64, -3.0, -1.5] {
        let edge_up = 1024.0 / x.abs().log2();
        let edge_down = -1075.0 / x.abs().log2();
        for edge in [edge_up, edge_down, -1022.0 / x.abs().log2()] {
            let start = edge.round() - 40.0;
            pairs.extend((0..80).map(|k| (x, start + k as f64)));
        }
    }
    check_pow(pairs, "negative base thresholds");
}

#[test]
fn test_pow_subnormal_results_match_mpfr() {
    // y = t / log2(x) for a target exponent t across the subnormal range, so the
    // result is near 2^t.
    let mut rng = SplitMix(0x5b);
    let mut pairs = Vec::with_capacity(60_000);
    for _ in 0..40_000 {
        let x = f64::from_bits(0x3fe0_0000_0000_0000 + rng.next_u64() % (3u64 << 52));
        let t = -1080.0 + 60.0 * (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let y = t / x.log2();
        if y.is_finite() {
            pairs.push((x, y));
        }
    }
    // Odd integer exponents give negative subnormal results for negative bases.
    for _ in 0..20_000 {
        let y = [3.0, 5.0, 7.0, 9.0, 11.0, 101.0][(rng.next_u64() % 6) as usize];
        let t = -1080.0 + 60.0 * (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let x = -(2f64.powf(t / y));
        pairs.push((x, y));
    }
    check_pow(pairs, "subnormal results");
}
