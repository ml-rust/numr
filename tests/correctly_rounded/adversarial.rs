//! Special values, thresholds and structured inputs against MPFR.

use super::oracle::{Function, check_all};

const HYDR_REGRESSION: f64 = 0.156_043_060_815_255_54;

/// `x` stepped `k` units in the last place along its bit pattern, both directions.
fn around(bits: u64, reach: u64) -> Vec<f64> {
    (0..=2 * reach)
        .filter_map(|k| bits.checked_add(k)?.checked_sub(reach))
        .map(f64::from_bits)
        .collect()
}

fn specials() -> Vec<f64> {
    let mut v = vec![
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7ff0_0000_0000_0001), // signalling NaN
        f64::from_bits(0xfff4_0000_0000_dead), // signalling NaN with payload
        f64::from_bits(0x7ff8_0000_0000_beef), // quiet NaN with payload
        f64::from_bits(0xffff_ffff_ffff_ffff),
        f64::from_bits(1),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::MIN_POSITIVE,
        f64::MAX,
        1.0,
        -1.0,
        2.0,
        0.5,
        HYDR_REGRESSION,
    ];
    let negatives: Vec<f64> = v.iter().map(|x| -x).collect();
    v.extend(negatives);
    v
}

#[test]
fn test_special_values_match_mpfr() {
    for function in Function::ALL {
        check_all(function, specials(), "special values");
    }
}

#[test]
fn test_special_values_give_canonical_nan() {
    for function in Function::ALL {
        for x in specials() {
            let y = function.numr(x);
            if y.is_nan() {
                assert_eq!(y.to_bits(), 0x7ff8_0000_0000_0000, "{function:?}({x:e})");
            }
        }
    }
}

#[test]
fn test_hydr_regression_input_matches_mpfr() {
    let bits = HYDR_REGRESSION.to_bits();
    for function in Function::ALL {
        check_all(function, around(bits, 64), "hydr regression neighbourhood");
        check_all(
            function,
            around((-HYDR_REGRESSION).to_bits(), 64),
            "negated regression",
        );
    }
}

#[test]
fn test_exp_thresholds_and_subnormal_band_match_mpfr() {
    let thresholds = [
        0x4086_2e42_fefa_39f0u64, // overflow
        0xc087_4910_d52d_3052,    // underflow to zero
        0xc086_232b_dd7a_bcd2,    // first subnormal result
        0x3c90_0000_0000_0000,    // 2^-54
        0xbc90_0000_0000_0000,    // -2^-54
    ];
    for bits in thresholds {
        check_all(Function::Exp, around(bits, 256), "threshold neighbourhood");
    }
    // Every result in [-745.14, -708.39] is subnormal.
    let (lo, hi) = (-745.14f64, -708.39f64);
    let n = 20_000;
    let band = (0..=n).map(|i| lo + (hi - lo) * (i as f64) / (n as f64));
    check_all(Function::Exp, band, "subnormal-output band");
}

#[test]
fn test_exp_of_powers_of_two_matches_mpfr() {
    let mut inputs = Vec::new();
    for k in 0..=1074 {
        let p = if k <= 1022 {
            2f64.powi(-k)
        } else {
            f64::from_bits(1u64 << (1074 - k))
        };
        for x in [p, -p] {
            inputs.extend(around(x.to_bits(), 2));
        }
    }
    for k in 1..10 {
        let p = 2f64.powi(k);
        inputs.extend([p, -p]);
    }
    check_all(Function::Exp, inputs, "exp of +-2^-k");
}

#[test]
fn test_log_near_one_and_powers_of_two_match_mpfr() {
    let one = 1f64.to_bits();
    for function in [Function::Log, Function::Log1p] {
        check_all(function, around(one, 4096), "near 1");
    }
    let mut powers = Vec::new();
    for bits in (1u64..=0x7fe).map(|e| e << 52) {
        powers.extend(around(bits, 3));
    }
    for k in 0..52 {
        powers.extend(around(1u64 << k, 1));
    }
    check_all(Function::Log, powers.clone(), "powers of two");
    check_all(Function::Log1p, powers, "powers of two");
    // The tiny-path bounds of log: 1 - 2^-9 and 1 + 2^-9.
    for bits in [0x3fef_f000_0000_0000u64, 0x3ff0_0800_0000_0000] {
        check_all(Function::Log, around(bits, 256), "tiny-path bounds");
    }
}

#[test]
fn test_log1p_near_minus_one_and_branch_bounds_match_mpfr() {
    let minus_one = (-1f64).to_bits();
    // Inputs just above -1, where 1 + x is tiny.
    let above: Vec<f64> = (1..=4096u64)
        .map(|k| f64::from_bits(minus_one - k))
        .collect();
    check_all(Function::Log1p, above, "near -1");
    check_all(Function::Log1p, around(minus_one, 16), "around -1");
    let bounds = [
        0x3ca0_0000_0000_0000u64, // 2^-53
        0x3f30_0000_0000_0000,    // 2^-12
        0x3ea1_9b00_0000_0000,    // 0x1.19bp-21
        0x3f50_0000_0000_0000,    // 2^-10
        0x3fb0_0000_0000_0000,    // 2^-4
        0x4340_0000_0000_0000,    // 2^53
        0x4690_0000_0000_0000,    // 2^106
        0x7fef_ffff_ffff_ffff,    // f64::MAX
        0x3fe0_0000_0000_0000,    // 0.5
    ];
    for bits in bounds {
        check_all(Function::Log1p, around(bits, 256), "branch bounds");
        check_all(
            Function::Log1p,
            around(bits | (1 << 63), 256),
            "negated branch bounds",
        );
    }
}
