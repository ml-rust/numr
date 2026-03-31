//! SIMD-accelerated activation functions.
//!
//! See [`dispatch`] for the public dispatch functions and scalar fallbacks.

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;
pub(crate) mod dispatch;

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64;

#[allow(unused_imports)]
pub use dispatch::*;
