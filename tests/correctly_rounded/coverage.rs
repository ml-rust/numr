//! Path coverage of the CORE-MATH hard-to-round fixtures in `tests/correctly_rounded/fixtures`.

use super::hard_cases::load;
use super::oracle::Function;
use numr::algorithm::correctly_rounded::traced::{Path, exp_traced, log_traced, log1p_traced};

/// Count of results per path: `[special, fast, accurate, exception]`.
fn paths(function: Function, kernel: fn(f64) -> (f64, Path)) -> [usize; 4] {
    let mut counts = [0; 4];
    for x in load(function) {
        let slot = match kernel(x).1 {
            Path::Special => 0,
            Path::Fast => 1,
            Path::Accurate => 2,
            Path::Exception => 3,
        };
        counts[slot] += 1;
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
