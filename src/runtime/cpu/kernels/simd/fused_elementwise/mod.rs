//! SIMD-accelerated fused elementwise operations.
//!
//! See [`dispatch`] for the public dispatch functions and scalar fallbacks.

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_64;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;

pub(crate) mod dispatch;

#[allow(unused_imports)]
pub use dispatch::*;
