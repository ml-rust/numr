//! Shared SIMD mathematical functions
//!
//! This module provides optimized SIMD implementations of transcendental functions
//! (exp, tanh, log, sin, cos, tan) that are used across multiple SIMD kernel modules.
//! By centralizing these implementations, we ensure consistency and eliminate code
//! duplication.
//!
//! # Supported Functions
//!
//! | Function | f32 | f64 | Algorithm |
//! |----------|-----|-----|-----------|
//! | exp      | ✓   | ✓   | Range reduction + Taylor series |
//! | tanh     | ✓   | ✓   | u/(u+2), u = expm1(2\|x\|) |
//! | log      | ✓   | ✓   | Exponent extraction + minimax polynomial |
//! | sin      | ✓   | ✓   | Range reduction + Taylor series |
//! | cos      | ✓   | ✓   | sin(x + π/2) |
//! | tan      | ✓   | ✓   | Range reduction + minimax polynomial |
//!
//! # Architecture Support
//!
//! | ISA | File | Vector Width | Notes |
//! |-----|------|--------------|-------|
//! | AVX2+FMA | avx2.rs | 256-bit | 8 f32 / 4 f64 |
//! | AVX-512F | avx512.rs | 512-bit | 16 f32 / 8 f64, native 64-bit ops |
//!
//! # Accuracy
//!
//! These approximations prioritize speed over full IEEE precision:
//! - Relative error: below 2 ulps for the exp, log and hyperbolic families in
//!   both precisions
//! - Valid input range: the whole representable domain for exp in both
//!   precisions, [-104, ln(f32::MAX)] and [-745, ln(f64::MAX)]
//! - Trigonometric functions: Accurate for |x| < 2^20; larger inputs may lose precision
//!
//! # Design Notes
//!
//! All polynomial coefficients are centralized in `common.rs` to ensure algorithm
//! consistency between AVX2 and AVX-512 implementations. The implementations differ
//! only in:
//! - Intrinsic prefixes (`_mm256` vs `_mm512`)
//! - Vector types (`__m256` vs `__m512`)
//! - Mask handling (AVX2 uses blendv, AVX-512 uses native masks)
//! - 64-bit operations (AVX-512 has native support, AVX2 requires workarounds)

pub mod common;

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
