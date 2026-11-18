//! `numr::algorithm::correctly_rounded` host functions against MPFR, bit for bit.
//!
//! The oracle is MPFR through `rug`: each function at 53-bit precision, rounded to
//! nearest, with binary64 underflow emulated and NaN canonical. Every output must equal
//! the oracle's bits.
//!
//! MSVC targets cannot build MPFR, so this binary is empty there.
#![cfg(not(target_env = "msvc"))]

#[path = "correctly_rounded/adversarial.rs"]
mod adversarial;
#[path = "correctly_rounded/coverage.rs"]
mod coverage;
#[path = "correctly_rounded/dense.rs"]
mod dense;
#[path = "correctly_rounded/hard_cases.rs"]
mod hard_cases;
#[path = "correctly_rounded/oracle.rs"]
mod oracle;
#[path = "correctly_rounded/pow_boundaries.rs"]
mod pow_boundaries;
#[path = "correctly_rounded/pow_dense.rs"]
mod pow_dense;
#[path = "correctly_rounded/pow_hard_cases.rs"]
mod pow_hard_cases;
#[path = "correctly_rounded/pow_special.rs"]
mod pow_special;
#[path = "correctly_rounded/thresholds.rs"]
mod thresholds;
