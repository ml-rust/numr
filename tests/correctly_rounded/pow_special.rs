//! `pow` special values, exact and midpoint cases, and hydr's exponents, against MPFR.

use super::oracle::{SplitMix, check_pow};
use numr::algorithm::correctly_rounded::pow;

const NAN: u64 = 0x7ff8_0000_0000_0000;

fn nans() -> [f64; 5] {
    [
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7ff0_0000_0000_0001), // signalling NaN
        f64::from_bits(0xfff4_0000_0000_dead), // signalling NaN with payload
        f64::from_bits(0x7ff8_0000_0000_beef), // quiet NaN with payload
    ]
}

/// Bases that hit every branch of Annex F, with both signs.
fn special_bases() -> Vec<f64> {
    let mut v = vec![
        0.0,
        f64::INFINITY,
        1.0,
        0.5,
        2.0,
        3.0,
        0.75,
        1.0 + f64::EPSILON,
        1.0 - f64::EPSILON / 2.0,
        f64::from_bits(1),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::MIN_POSITIVE,
        f64::MAX,
        10.0,
    ];
    let negatives: Vec<f64> = v.iter().map(|x| -x).collect();
    v.extend(negatives);
    v.extend(nans());
    v
}

/// Exponents covering zeros, infinities, odd and even integers, and non-integers.
fn special_exponents() -> Vec<f64> {
    let mut v = vec![
        0.0,
        f64::INFINITY,
        1.0,
        2.0,
        3.0,
        0.5,
        1.5,
        2.5,
        1.0e-300,
        f64::from_bits(1),
        f64::MAX,
        9_007_199_254_740_991.0, // 2^53 - 1, odd
        9_007_199_254_740_992.0, // 2^53, even
        9_007_199_254_740_994.0, // 2^53 + 2, even
        4_503_599_627_370_497.0, // 2^52 + 1, odd
        4_503_599_627_370_496.5, // not an integer
        1.0e300,
        1075.0,
        1074.0,
        1023.0,
        1024.0,
    ];
    let negatives: Vec<f64> = v.iter().map(|y| -y).collect();
    v.extend(negatives);
    v.extend(nans());
    v
}

fn grid() -> Vec<(f64, f64)> {
    let ys = special_exponents();
    special_bases()
        .into_iter()
        .flat_map(|x| ys.iter().map(move |&y| (x, y)))
        .collect()
}

#[test]
fn test_pow_special_grid_matches_mpfr() {
    check_pow(grid(), "special-value grid");
}

#[test]
fn test_pow_special_grid_gives_canonical_nan() {
    for (x, y) in grid() {
        let r = pow(x, y);
        if r.is_nan() {
            assert_eq!(r.to_bits(), NAN, "pow({x:e}, {y:e})");
        }
    }
}

/// The Annex F table as documented on `pow`, checked without the oracle.
#[test]
fn test_pow_annex_f_table() {
    let inf = f64::INFINITY;
    let rows: &[(f64, f64, f64)] = &[
        (f64::NAN, 0.0, 1.0),
        (f64::NAN, -0.0, 1.0),
        (f64::from_bits(0x7ff0_0000_0000_0001), 0.0, 1.0),
        (1.0, f64::NAN, 1.0),
        (1.0, f64::from_bits(0x7ff0_0000_0000_0001), 1.0),
        (1.0, -inf, 1.0),
        (-1.0, inf, 1.0),
        (-1.0, -inf, 1.0),
        (0.0, -3.0, inf),
        (-0.0, -3.0, -inf),
        (0.0, -2.0, inf),
        (-0.0, -2.0, inf),
        (-0.0, -0.5, inf),
        (-0.0, -inf, inf),
        (0.0, 3.0, 0.0),
        (-0.0, 3.0, -0.0),
        (-0.0, 2.0, 0.0),
        (-0.0, 0.5, 0.0),
        (-0.0, inf, 0.0),
        (0.5, -inf, inf),
        (-0.5, -inf, inf),
        (2.0, -inf, 0.0),
        (-2.0, -inf, 0.0),
        (0.5, inf, 0.0),
        (-0.5, inf, 0.0),
        (2.0, inf, inf),
        (-2.0, inf, inf),
        (-inf, -3.0, -0.0),
        (-inf, -2.0, 0.0),
        (-inf, -0.5, 0.0),
        (-inf, 3.0, -inf),
        (-inf, 2.0, inf),
        (-inf, 0.5, inf),
        (inf, -0.5, 0.0),
        (inf, 0.5, inf),
        (-2.0, 3.0, -8.0),
        (-2.0, 2.0, 4.0),
        (-2.0, -1.0, -0.5),
    ];
    for &(x, y, want) in rows {
        assert_eq!(pow(x, y).to_bits(), want.to_bits(), "pow({x:e}, {y:e})");
    }
    for (x, y) in [(-2.0, 0.5), (-8.0, 1.0 / 3.0), (-1.0, 2.5), (f64::NAN, 1.0)] {
        assert_eq!(pow(x, y).to_bits(), NAN, "pow({x:e}, {y:e})");
    }
    assert_eq!(pow(2.0, f64::NAN).to_bits(), NAN);
}

/// `2^k` for every binary64 power of two.
fn powers_of_two() -> impl Iterator<Item = f64> {
    (0u64..=0x7fe)
        .map(|e| f64::from_bits(e << 52))
        .filter(|&x| x != 0.0)
        .chain((0..52).map(|k| f64::from_bits(1u64 << k)))
}

#[test]
fn test_pow_of_powers_of_two_matches_mpfr() {
    let ys = [
        -3.0,
        -2.0,
        -1.0,
        2.0,
        3.0,
        5.0,
        34.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        0.25,
        0.125,
        0.03125,
        0.75,
        2.5,
        1.0e-3,
        1.0 / 3.0,
        1.1,
    ];
    let pairs = powers_of_two().flat_map(|x| ys.iter().flat_map(move |&y| [(x, y), (-x, y)]));
    check_pow(pairs, "powers of two");
}

#[test]
fn test_pow_integer_exponents_with_exact_and_midpoint_results_match_mpfr() {
    // m^y for odd m runs past 53 bits: exact results, 54-bit midpoints, and inexact ones.
    let mut pairs = Vec::new();
    for y in 2..=34u32 {
        // Small odd bases, then odd bases around the one whose power reaches 2^53.
        let edge = (2f64.powf(53.0 / y as f64) as u64) | 1;
        let small = (3..=199u64).step_by(2);
        let near = (edge.saturating_sub(400).max(3) | 1..=edge + 400).step_by(2);
        for m in small.chain(near) {
            for e in [
                0i32,
                7,
                -9,
                -1074 / y as i32,
                1023 / y as i32 - 1,
                -1074 / y as i32 - 3,
            ] {
                let x = m as f64 * 2f64.powi(e);
                pairs.push((x, y as f64));
                pairs.push((-x, y as f64));
            }
        }
    }
    check_pow(pairs, "integer exponents");
}

#[test]
fn test_pow_fractional_exponents_with_exact_results_match_mpfr() {
    // x = s^q * 2^(q e) with y = n / q: exact and midpoint results, subnormal ones included.
    let mut rng = SplitMix(0x9e11);
    let mut pairs = Vec::new();
    let exponents: [(f64, u32); 7] = [
        (0.5, 2),
        (1.5, 2),
        (2.5, 2),
        (-0.5, 2),
        (0.25, 4),
        (0.75, 4),
        (0.125, 8),
    ];
    for (y, q) in exponents {
        for _ in 0..2_000 {
            let bits = 53 / q;
            let s = (rng.next_u64() >> (64 - bits)) | 1;
            let base = (s as f64).powi(q as i32);
            let e = (rng.next_u64() % 2_000) as i32 - 1_000;
            let scale = e - e.rem_euclid(q as i32);
            let x = base * 2f64.powi(scale);
            if x > 0.0 && x.is_finite() {
                pairs.push((x, y));
            }
        }
    }
    check_pow(pairs, "fractional exponents");
}

/// The doubles nearest the exponents hydr uses, and neighbours of the exact cases.
pub fn hydr_exponents() -> Vec<f64> {
    vec![
        4.0 / 3.0,
        7.0 / 3.0,
        5.0 / 3.0,
        1.666_666_7,
        1.0 - 1.666_666_7,
        1.5,
        0.385,
        0.5,
        1.0 / 3.0,
        -1.0,
        2.0,
        3.0,
        1.0e6,
        4_503_599_627_370_497.0,
    ]
}

#[test]
fn test_pow_hydr_exponents_over_every_binade_match_mpfr() {
    let mut rng = SplitMix(0x4d2);
    let mut pairs = Vec::new();
    for y in hydr_exponents() {
        for exponent in 0u64..=0x7fe {
            for _ in 0..2 {
                let x = f64::from_bits((exponent << 52) | (rng.next_u64() >> 12));
                pairs.push((x, y));
                if y.fract() == 0.0 {
                    pairs.push((-x, y));
                }
            }
        }
        pairs.push((1.0, y));
        pairs.push((0.304_8, y));
    }
    check_pow(pairs, "hydr exponents");
}
