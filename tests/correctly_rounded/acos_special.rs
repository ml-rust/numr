//! `acos` special values, branch thresholds and the listed exceptions against MPFR.

use super::adversarial::specials;
use super::oracle::check_acos;
use numr::algorithm::correctly_rounded::traced::{Path, acos_traced};
use numr::algorithm::correctly_rounded::{self as cr, CANONICAL_NAN};

/// `bits` stepped `k` units in the last place along its bit pattern, `0 <= k <= reach`, both directions.
fn around(bits: u64, reach: u64) -> Vec<f64> {
    (0..=2 * reach)
        .filter_map(|k| bits.checked_add(k)?.checked_sub(reach))
        .map(f64::from_bits)
        .collect()
}

/// `x` and `-x` for every `x` in `values`.
fn both_signs(values: Vec<f64>) -> Vec<f64> {
    let negated: Vec<f64> = values.iter().map(|x| -x).collect();
    values.into_iter().chain(negated).collect()
}

#[test]
fn test_acos_special_values_match_mpfr() {
    check_acos(specials(), "special values");
    let domain_edge = vec![
        1.0,
        -1.0,
        1.0 + f64::EPSILON,
        -1.0 - f64::EPSILON,
        2.0,
        -1.0e300,
        f64::MAX,
        f64::MIN,
    ];
    check_acos(domain_edge, "domain edge");
}

#[test]
fn test_acos_nan_results_are_canonical() {
    let mut inputs = specials();
    inputs.extend([1.5, -1.5, 1.0 + f64::EPSILON, f64::MAX]);
    for x in inputs {
        let y = cr::acos(x);
        if x.is_nan() || x.abs() > 1.0 {
            assert_eq!(y.to_bits(), CANONICAL_NAN.to_bits(), "acos({x:e})");
            assert_eq!(acos_traced(x).1, Path::Special, "acos({x:e})");
        } else {
            assert!(!y.is_nan(), "acos({x:e})");
        }
    }
}

#[test]
fn test_acos_of_zeros_and_subnormals_matches_mpfr() {
    let mut inputs = vec![0.0, -0.0, f64::MIN_POSITIVE];
    for k in 0..52 {
        inputs.extend(around(1u64 << k, 1));
    }
    inputs.extend(around(0x000f_ffff_ffff_ffff, 4));
    check_acos(both_signs(inputs), "zeros and subnormals");
}

#[test]
fn test_acos_branch_thresholds_match_mpfr() {
    let thresholds = [
        0x3fe0_0000_0000_0000u64, // 0.5: table path up to it, double angle above
        0x3f00_0000_0000_0000,    // 2^-15: tiny path up to it
        0x3c8c_b3b3_8697_47f4,    // 0x1.cb3b3869747f4p-55: cubic term from above it
        0x3fb0_0000_0000_0000,    // 2^-4: start of the exhaustively searched range
        0x3ff0_0000_0000_0000,    // 1
    ];
    for bits in thresholds {
        check_acos(both_signs(around(bits, 256)), &format!("around {bits:#x}"));
    }
    // 1 - k ulp and -1 + k ulp, where the result is smallest and the double-angle
    // reduction carries most of the error.
    let near_one: Vec<f64> = (1..=4096u64)
        .map(|k| f64::from_bits(1f64.to_bits() - k))
        .collect();
    check_acos(both_signs(near_one), "near +-1");
    // Row boundaries of the asin table: x^2 = (j + 1/2)/128 on the table path and
    // 2 - 2|x| = (j + 1/2)/32 on the double-angle path.
    let mut rows = Vec::new();
    for j in 0..32 {
        let mid = (j as f64 + 0.5) / 128.0;
        rows.extend(around(mid.sqrt().to_bits(), 8));
        let w = (j as f64 + 0.5) / 32.0;
        rows.extend(around((1.0 - w / 2.0).to_bits(), 8));
    }
    check_acos(both_signs(rows), "table row boundaries");
}

#[test]
fn test_acos_listed_exceptions_match_mpfr() {
    let listed: [u64; 5] = [
        0x3fef_ffff_ffff_fdc0,
        0x3fb5_3ea6_c725_5e88,
        0x3f4f_d737_be91_4578,
        0x3fef_ffff_ffff_ff70,
        0x3fa3_90e6_939c_d1a6,
    ];
    // Under round-to-nearest without contraction, only 0x1.53ea6c7255e88p-4 fails the
    // fast-path test and reaches the exception table.
    let expected = [
        Path::Fast,
        Path::Exception,
        Path::Fast,
        Path::Fast,
        Path::Fast,
    ];
    for (bits, path) in listed.into_iter().zip(expected) {
        let x = f64::from_bits(bits);
        assert_eq!(acos_traced(x).1, path, "acos({x:e})");
    }
    let inputs: Vec<f64> = listed.iter().flat_map(|&b| around(b, 16)).collect();
    check_acos(both_signs(inputs), "listed exceptions");
}
