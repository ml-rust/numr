//! Correctly rounded binary64 elementary functions.
//!
//! - `exp`, `log`, `log1p`: each result is the infinitely precise value rounded to nearest,
//!   ties to even, with gradual underflow.
//! - `pow`: the same for every input pair whose final 256-bit rounding test succeeds.
//!   CORE-MATH proves correct rounding of `pow` modulo the hardest-to-round cases, which are
//!   unknown for binary64 (Hubrecht, Jeannerod and Zimmermann, ARITH 2023,
//!   <https://inria.hal.science/hal-04159652>). No known input fails the test. If one does,
//!   the result is the 256-bit approximation rounded to nearest.
//!
//! The bits are therefore a property of the mathematical function: they do not depend on
//! the platform libm, the target, or the caller.
//!
//! The kernels port CORE-MATH (MIT licence, <https://core-math.gitlabpages.inria.fr/>).
//! They use only IEEE add, subtract, multiply and `f64::mul_add`, in a fixed order, plus
//! exact integer operations on bit patterns. The caller keeps round-to-nearest-even and
//! gradual underflow active. Every NaN result is the canonical quiet NaN
//! `0x7ff8000000000000`. IEEE exception flags are not promised.

mod dint;
mod double_double;
mod exp;
mod log;
mod log1p;
mod log1p_refine;
mod log_accurate;
mod path;
mod pow;
mod pow_accurate;
mod pow_dint;
mod pow_exact;
mod pow_extended;
mod pow_fast;
mod qint;
mod qint_mul;
mod tables;

#[doc(hidden)]
pub mod traced;

pub use double_double::CANONICAL_NAN;
pub use exp::exp;
pub use log::log;
pub use log1p::log1p;
pub use pow::pow;
