//! Correctly rounded binary64 elementary functions.
//!
//! Each function returns the infinitely precise result rounded to nearest, ties to even,
//! with gradual underflow. The bits are therefore a property of the mathematical function:
//! they do not depend on the platform libm, the target, or the caller.
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
mod tables;

#[doc(hidden)]
pub mod traced;

pub use double_double::CANONICAL_NAN;
pub use exp::exp;
pub use log::log;
pub use log1p::log1p;
