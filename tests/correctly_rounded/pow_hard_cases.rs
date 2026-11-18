//! CORE-MATH hard-to-round `pow` pairs against MPFR.
//!
//! `fixtures/pow.bin` comes from `src/binary64/pow/pow.wc` of CORE-MATH
//! (<https://gitlab.inria.fr/core-math/core-math>, MIT licence), commit
//! 1851882b0f85ed581af9185cfa020992cec9e7a1, through
//! `scripts/correctly_rounded/port_hard_cases.py`. It groups the distinct finite pairs by
//! the bit pattern of `y`: per group, the LEB128 delta of `y`, the LEB128 pair count, then
//! the sorted `x` bit patterns as LEB128 deltas. Pairs with a non-finite operand are left out.

use super::oracle::check_pow;

/// The default run checks one in `DEFAULT_STRIDE` pairs; the ignored test checks every pair.
const DEFAULT_STRIDE: usize = 16;

/// Decode the fixture into its pairs.
pub fn load_pow() -> Vec<(f64, f64)> {
    let path = format!(
        "{}/tests/correctly_rounded/fixtures/pow.bin",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = std::fs::read(&path).unwrap_or_else(|err| {
        panic!("read {path}: {err}. Rerun scripts/correctly_rounded/port_hard_cases.py")
    });
    let mut pos = 0;
    let mut pairs = Vec::new();
    let mut y = 0u64;
    while pos < bytes.len() {
        y = y.wrapping_add(leb128(&bytes, &mut pos, &path));
        let count = leb128(&bytes, &mut pos, &path);
        let mut x = 0u64;
        for _ in 0..count {
            x = x.wrapping_add(leb128(&bytes, &mut pos, &path));
            pairs.push((f64::from_bits(x), f64::from_bits(y)));
        }
    }
    pairs
}

/// The unsigned LEB128 value at `pos`, advancing `pos` past it.
fn leb128(bytes: &[u8], pos: &mut usize, path: &str) -> u64 {
    let (mut value, mut shift) = (0u64, 0u32);
    loop {
        let byte = *bytes
            .get(*pos)
            .unwrap_or_else(|| panic!("{path}: truncated LEB128 value at byte {pos}"));
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return value;
        }
        shift += 7;
    }
}

fn run(stride: usize) {
    let pairs = load_pow();
    assert!(
        pairs.len() > 900_000,
        "pow fixture holds {} pairs",
        pairs.len()
    );
    check_pow(pairs.into_iter().step_by(stride), "hard cases");
}

#[test]
fn test_pow_hard_cases_sample_match_mpfr() {
    run(DEFAULT_STRIDE);
}

#[test]
#[ignore = "every CORE-MATH pow hard case; run with: cargo test --test correctly_rounded -- --ignored"]
fn test_all_pow_hard_cases_match_mpfr() {
    run(1);
}
