//! CORE-MATH `acos` hard-to-round inputs and fixed-seed samples against MPFR, and the
//! paths they reach.
//!
//! `fixtures/acos.bin` comes from `src/binary64/acos/acos.wc` of CORE-MATH
//! (<https://gitlab.inria.fr/core-math/core-math>, MIT licence), commit
//! 1851882b0f85ed581af9185cfa020992cec9e7a1, through
//! `scripts/correctly_rounded/port_hard_cases.py`, in the layout of the other fixtures.

use super::hard_cases::load_fixture;
use super::oracle::{SplitMix, check_acos};
use numr::algorithm::correctly_rounded::traced::{Path, acos_traced};

/// Branch of the fast path that `|x|` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Branch {
    Tiny,
    Table,
    DoubleAngle,
}

fn branch(x: f64) -> Branch {
    let a = x.abs();
    if a <= f64::from_bits(0x3f00_0000_0000_0000) {
        Branch::Tiny
    } else if a <= 0.5 {
        Branch::Table
    } else {
        Branch::DoubleAngle
    }
}

fn hard_cases() -> Vec<f64> {
    let inputs = load_fixture("acos");
    assert!(
        inputs.len() > 200_000,
        "acos fixture holds {} inputs",
        inputs.len()
    );
    inputs
}

#[test]
fn test_all_acos_hard_cases_match_mpfr() {
    check_acos(hard_cases(), "hard cases");
}

#[test]
fn test_acos_hard_cases_reach_the_accurate_path_from_every_branch() {
    let mut seen = Vec::new();
    for x in hard_cases() {
        let (_, path) = acos_traced(x);
        if path == Path::Accurate && !seen.contains(&branch(x)) {
            seen.push(branch(x));
        }
        // The list also holds +-1 and inputs above 1 in magnitude.
        assert_eq!(
            path == Path::Special,
            x.abs() >= 1.0,
            "acos({x:e}) took {path:?}"
        );
    }
    for b in [Branch::Tiny, Branch::Table, Branch::DoubleAngle] {
        assert!(seen.contains(&b), "{b:?} never reaches the accurate path");
    }
}

#[test]
fn test_acos_paths_cover_fast_accurate_exception_and_special() {
    let mut counts = [0usize; 4];
    let mut inputs = hard_cases();
    inputs.extend([2.0, f64::NAN, -1.0, f64::from_bits(0x3fb5_3ea6_c725_5e88)]);
    for x in inputs {
        let slot = match acos_traced(x).1 {
            Path::Special => 0,
            Path::Fast => 1,
            Path::Accurate => 2,
            Path::Exception => 3,
            other => panic!("acos({x:e}) took {other:?}"),
        };
        counts[slot] += 1;
    }
    assert!(counts.iter().all(|&n| n > 0), "acos paths {counts:?}");
}

/// `count` inputs of `[lo, hi]` (`0 <= lo < hi`) with uniformly random bit patterns,
/// each with a random sign.
fn sample(rng: &mut SplitMix, lo: f64, hi: f64, count: usize) -> Vec<f64> {
    let (a, b) = (lo.to_bits(), hi.to_bits());
    (0..count)
        .map(|_| {
            let r = rng.next_u64();
            let x = f64::from_bits(a + (r >> 1) % (b - a + 1));
            if r & 1 == 1 { -x } else { x }
        })
        .collect()
}

#[test]
fn test_acos_binade_and_range_samples_match_mpfr() {
    let mut rng = SplitMix(0xac05_0001);
    // 32 random significands in every binade of [2^-1074, 1), both signs. Binades below
    // 2^-4 hold 1018 of the 1023, so the sample leans toward the unproven band.
    let mut inputs = Vec::new();
    for exponent in 0u64..0x3ff {
        for _ in 0..32 {
            let r = rng.next_u64();
            let bits = (exponent << 52) | (r & 0x000f_ffff_ffff_ffff) | (r & (1 << 63));
            inputs.push(f64::from_bits(bits));
        }
    }
    check_acos(inputs, "binade sample");
    // The unproven band, where hydr's culvert geometry lands, and the proven range.
    check_acos(sample(&mut rng, 0.0, 0.0625, 40_000), "|x| < 2^-4");
    check_acos(sample(&mut rng, 0.0625, 1.0, 40_000), "2^-4 <= |x| <= 1");
    // Uniformly random bit patterns: mostly |x| > 1 and NaN, which must give NaN.
    check_acos(
        (0..5_000).map(|_| f64::from_bits(rng.next_u64())),
        "random bits",
    );
}

#[test]
fn test_acos_dense_sweep_matches_mpfr() {
    let mut rng = SplitMix(0xac05_0002);
    check_acos(sample(&mut rng, 0.0, 0.0625, 2_000_000), "|x| < 2^-4 sweep");
    check_acos(
        sample(&mut rng, 0.0625, 1.0, 2_000_000),
        "2^-4 <= |x| <= 1 sweep",
    );
}
