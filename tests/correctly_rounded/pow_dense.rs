//! Fixed-seed `pow` samples over the domain and over raw bit patterns, against MPFR.

use super::oracle::{SplitMix, check_pow};
use super::pow_special::hydr_exponents;

/// A uniform value in `[0, 1)` from 53 random bits.
fn unit(rng: &mut SplitMix) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// Pairs with `x` uniform over the bit patterns of positive finite doubles and `y` chosen
/// so that `|y log x|` spans the finite range, slightly past both ends.
fn domain(seed: u64, count: usize) -> Vec<(f64, f64)> {
    let mut rng = SplitMix(seed);
    let mut pairs = Vec::with_capacity(count);
    while pairs.len() < count {
        let x = f64::from_bits(1 + rng.next_u64() % 0x7fef_ffff_ffff_ffff);
        let ln = x.ln();
        if ln == 0.0 {
            continue;
        }
        let target = (2.0 * unit(&mut rng) - 1.0) * 760.0;
        pairs.push((x, target / ln));
    }
    pairs
}

/// Pairs with both operands uniform over all bit patterns.
fn raw(seed: u64, count: usize) -> Vec<(f64, f64)> {
    let mut rng = SplitMix(seed);
    (0..count)
        .map(|_| {
            (
                f64::from_bits(rng.next_u64()),
                f64::from_bits(rng.next_u64()),
            )
        })
        .collect()
}

/// Pairs with `x` near 1, where `log x` cancels, and `y` large.
fn near_one(seed: u64, count: usize) -> Vec<(f64, f64)> {
    let mut rng = SplitMix(seed);
    (0..count)
        .map(|_| {
            let k = (rng.next_u64() % (1 << 30)) as i64 - (1 << 29);
            let x = f64::from_bits((1f64.to_bits() as i64 + k) as u64);
            let y = (2.0 * unit(&mut rng) - 1.0) * 2f64.powi((rng.next_u64() % 64) as i32);
            (x, y)
        })
        .collect()
}

/// Pairs with every hydr exponent and `x` from random bit patterns across every binade.
fn binades(seed: u64, per_binade: usize) -> Vec<(f64, f64)> {
    let mut rng = SplitMix(seed);
    let mut pairs = Vec::new();
    for y in hydr_exponents() {
        for exponent in 0u64..=0x7fe {
            for _ in 0..per_binade {
                pairs.push((f64::from_bits((exponent << 52) | (rng.next_u64() >> 12)), y));
            }
        }
    }
    pairs
}

#[test]
fn test_pow_domain_samples_match_mpfr() {
    check_pow(domain(0x70, 40_000), "domain sample");
    check_pow(near_one(0x71, 20_000), "near one");
}

#[test]
fn test_pow_raw_bit_patterns_match_mpfr() {
    check_pow(raw(0x72, 20_000), "raw bit patterns");
}

#[test]
#[ignore = "dense sweep; run with: cargo test --test correctly_rounded -- --ignored"]
fn test_pow_dense_sweep_matches_mpfr() {
    check_pow(domain(0x80, 2_000_000), "domain sweep");
    check_pow(near_one(0x81, 500_000), "near one sweep");
    check_pow(raw(0x82, 500_000), "raw sweep");
    check_pow(binades(0x83, 64), "hydr exponent sweep");
}
