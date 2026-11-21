//! CORE-MATH hard-to-round inputs against MPFR.
//!
//! `fixtures/{exp,log,log1p}.bin` come from `src/binary64/{f}/{f}.wc` of CORE-MATH
//! (<https://gitlab.inria.fr/core-math/core-math>, MIT licence), commit
//! 1851882b0f85ed581af9185cfa020992cec9e7a1, through
//! `scripts/correctly_rounded/port_hard_cases.py`. Each file holds sorted distinct
//! binary64 bit patterns as unsigned LEB128 deltas. Non-finite entries are left out.

use super::oracle::{Function, check_all};

/// Every input of the default run is one in `DEFAULT_STRIDE` of each list; the ignored
/// test runs every input.
const DEFAULT_STRIDE: usize = 16;

/// Decode one fixture into its inputs.
pub fn load(function: Function) -> Vec<f64> {
    let name = match function {
        Function::Exp => "exp",
        Function::Log => "log",
        Function::Log1p => "log1p",
    };
    load_fixture(name)
}

/// Decode `fixtures/{name}.bin` into its inputs.
pub fn load_fixture(name: &str) -> Vec<f64> {
    let path = format!(
        "{}/tests/correctly_rounded/fixtures/{name}.bin",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|err| {
        panic!("read {path}: {err}. Rerun scripts/correctly_rounded/port_hard_cases.py")
    });
    let mut values = Vec::new();
    let (mut previous, mut delta, mut shift) = (0u64, 0u64, 0u32);
    for byte in bytes {
        delta |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            previous = previous.wrapping_add(delta);
            values.push(f64::from_bits(previous));
            delta = 0;
            shift = 0;
        } else {
            shift += 7;
        }
    }
    assert_eq!(shift, 0, "{path}: truncated LEB128 delta");
    values
}

fn run(function: Function, stride: usize) {
    let inputs = load(function);
    assert!(
        inputs.len() > 100_000,
        "{function:?}: fixture holds {} inputs",
        inputs.len()
    );
    check_all(function, inputs.into_iter().step_by(stride), "hard cases");
}

#[test]
fn test_exp_hard_cases_sample_match_mpfr() {
    run(Function::Exp, DEFAULT_STRIDE);
}

#[test]
fn test_log_hard_cases_sample_match_mpfr() {
    run(Function::Log, DEFAULT_STRIDE);
}

#[test]
fn test_log1p_hard_cases_sample_match_mpfr() {
    run(Function::Log1p, DEFAULT_STRIDE);
}

#[test]
#[ignore = "every CORE-MATH hard case; run with: cargo test --test correctly_rounded -- --ignored"]
fn test_all_hard_cases_match_mpfr() {
    for function in Function::ALL {
        run(function, 1);
    }
}
