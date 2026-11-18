//! Fixed-seed samples over every binade and over raw bit patterns, against MPFR.

use super::oracle::{Function, SplitMix, check_all};

/// Inputs with `per_binade` random significands in every binade of both signs,
/// subnormals included, then `random` uniformly random bit patterns.
fn sample(seed: u64, per_binade: usize, random: usize) -> Vec<f64> {
    let mut rng = SplitMix(seed);
    let mut inputs = Vec::with_capacity(2 * 2047 * per_binade + random);
    for sign in [0u64, 1 << 63] {
        for exponent in 0u64..=0x7fe {
            for _ in 0..per_binade {
                let fraction = rng.next_u64() & 0x000f_ffff_ffff_ffff;
                inputs.push(f64::from_bits(sign | (exponent << 52) | fraction));
            }
        }
    }
    inputs.extend((0..random).map(|_| f64::from_bits(rng.next_u64())));
    inputs
}

/// Inputs in `[lo, hi]` with uniformly random bit patterns. A range that spans zero is
/// sampled as its negative and positive halves, `count / 2` inputs each.
fn sample_range(seed: u64, lo: f64, hi: f64, count: usize) -> Vec<f64> {
    if lo < 0.0 && hi > 0.0 {
        let mut inputs = sample_range(seed, lo, -0.0, count / 2);
        inputs.extend(sample_range(seed ^ 0xa5a5, 0.0, hi, count - count / 2));
        return inputs;
    }
    let mut rng = SplitMix(seed);
    let (a, b) = (lo.to_bits(), hi.to_bits());
    let (a, b) = (a.min(b), a.max(b));
    (0..count)
        .map(|_| f64::from_bits(a + rng.next_u64() % (b - a + 1)))
        .collect()
}

#[test]
fn test_binade_and_random_samples_match_mpfr() {
    for (i, function) in Function::ALL.into_iter().enumerate() {
        check_all(
            function,
            sample(0x5eed_0000 + i as u64, 4, 20_000),
            "dense sample",
        );
    }
}

#[test]
fn test_domain_samples_match_mpfr() {
    // The ranges where each function returns a finite, nonzero, non-trivial result.
    check_all(
        Function::Exp,
        sample_range(11, -745.2, 709.8, 40_000),
        "exp domain",
    );
    check_all(
        Function::Exp,
        sample_range(12, -1.0, 1.0, 20_000),
        "exp near zero",
    );
    check_all(
        Function::Log,
        sample_range(13, 0.5, 2.0, 40_000),
        "log near one",
    );
    check_all(
        Function::Log1p,
        sample_range(14, -0.999, 1.0, 40_000),
        "log1p domain",
    );
    check_all(
        Function::Log1p,
        sample_range(15, -0.0625, 0.0625, 20_000),
        "log1p near zero",
    );
}

#[test]
#[ignore = "dense sweep; run with: cargo test --test correctly_rounded -- --ignored"]
fn test_dense_sweep_matches_mpfr() {
    for (i, function) in Function::ALL.into_iter().enumerate() {
        check_all(
            function,
            sample(0xd0_0000 + i as u64, 256, 2_000_000),
            "dense sweep",
        );
    }
    check_all(
        Function::Exp,
        sample_range(21, -745.2, 709.8, 2_000_000),
        "exp domain sweep",
    );
    check_all(
        Function::Log,
        sample_range(22, 0.5, 2.0, 2_000_000),
        "log near one sweep",
    );
    check_all(
        Function::Log1p,
        sample_range(23, -0.999, 1.0, 2_000_000),
        "log1p domain sweep",
    );
}
