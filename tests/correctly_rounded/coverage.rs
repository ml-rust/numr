//! Path coverage of the CORE-MATH hard-to-round fixtures in `tests/correctly_rounded/fixtures`.

use super::hard_cases::load;
use super::oracle::Function;
use super::pow_hard_cases::load_pow;
use numr::algorithm::correctly_rounded::traced::{
    Path, exp_traced, log_traced, log1p_traced, pow_traced,
};

/// Slot of a path in a count array.
fn slot(path: Path) -> usize {
    match path {
        Path::Special => 0,
        Path::Fast => 1,
        Path::Accurate => 2,
        Path::Exception => 3,
        Path::Exact => 4,
        Path::Extended => 5,
        Path::Uncertified => 6,
    }
}

/// Count of results per path, in [`slot`] order.
fn paths(function: Function, kernel: fn(f64) -> (f64, Path)) -> [usize; 7] {
    let mut counts = [0; 7];
    for x in load(function) {
        counts[slot(kernel(x).1)] += 1;
    }
    counts
}

#[test]
fn test_hard_cases_reach_the_accurate_path_of_every_function() {
    let exp = paths(Function::Exp, exp_traced);
    assert!(exp[2] > 0, "exp paths {exp:?}");
    assert!(exp[3] > 0, "exp paths {exp:?}");
    let log = paths(Function::Log, log_traced);
    assert!(log[2] > 0, "log paths {log:?}");
    let log1p = paths(Function::Log1p, log1p_traced);
    assert!(log1p[2] > 0, "log1p paths {log1p:?}");
    // The fixtures exercise the fast path too, so the MPFR run covers both.
    for (name, counts) in [("exp", exp), ("log", log), ("log1p", log1p)] {
        assert!(counts[1] > 0, "{name} paths {counts:?}");
    }
}

#[test]
fn test_pow_hard_cases_reach_every_pow_path() {
    let mut counts = [0usize; 7];
    for (x, y) in load_pow() {
        counts[slot(pow_traced(x, y).1)] += 1;
    }
    // Fast, accurate, exact and extended paths all run; the uncertified branch never does.
    for path in [Path::Fast, Path::Accurate, Path::Exact, Path::Extended] {
        assert!(
            counts[slot(path)] > 0,
            "{path:?} unreached, pow paths {counts:?}"
        );
    }
    assert_eq!(counts[slot(Path::Uncertified)], 0, "pow paths {counts:?}");
}

#[test]
fn test_pow_tiny_exponents_reach_the_extended_path() {
    // |y log x| < 2^-55 fails both earlier rounding tests and resolves to 1 there.
    for (x, y) in [
        (3.0, 2f64.powi(-1000)),
        (0.5, -1.0e-300),
        (1.0e300, 5.0e-324),
    ] {
        assert_eq!(pow_traced(x, y), (1.0, Path::Extended), "pow({x:e}, {y:e})");
    }
}
