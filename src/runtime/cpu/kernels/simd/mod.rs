//! SIMD detection and dispatch for CPU kernels
//!
//! This module provides runtime CPU feature detection. Each operation type
//! (matmul, binary, unary, reduce) has its own submodule with SIMD kernels.
//!
//! # Module Structure
//!
//! ```text
//! simd/
//! ├── mod.rs              # This file: detection only
//! ├── {operation}/        # Each operation type
//! │   ├── mod.rs          # Unified dispatch (handles all architectures)
//! │   ├── x86_64/         # x86-64 implementations
//! │   │   ├── mod.rs
//! │   │   ├── avx2.rs
//! │   │   └── avx512.rs
//! │   └── aarch64/        # ARM64 implementations
//! │       ├── mod.rs
//! │       └── neon.rs
//! ```
//!
//! # Architecture Support
//!
//! | Architecture | Instruction Set | Vector Width | Status    |
//! |--------------|-----------------|--------------|-----------|
//! | x86-64       | AVX-512F/VL/DQ/BW + FMA | 512 bits | Supported |
//! | x86-64       | AVX2 + FMA      | 256 bits     | Supported |
//! | ARM64        | NEON + FP16     | 128 bits     | Supported |
//! | ARM64        | NEON            | 128 bits     | Supported |
//! | Any          | Scalar          | N/A          | Fallback  |

// Shared f16/bf16 ↔ f32 SIMD conversion utilities
#[cfg(all(feature = "f16", any(target_arch = "x86_64", target_arch = "aarch64")))]
pub mod half_convert_utils;

// Macros for generating f16/bf16 block-convert-compute wrappers (must come before users)
// Compiled on every SIMD target. The macros gate generated code with #[cfg(feature = "f16")].
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[macro_use]
mod half_macros;

// CPU SIMD capability detection
mod detect;

// Operation modules. Each operation's mod.rs handles internal architecture dispatch.
// Callers reach the elementwise, reduction, norm and conv modules only on SIMD
// targets, so those modules compile only there. Other targets run generic loops.
// `distance`, `dot` and `matmul` stay on every target: they hold arch-neutral entry points.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod activations;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod binary;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod clamp;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod compare;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod conv;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod cumulative;
pub mod distance;
pub mod dot;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod fused_activation_mul;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod fused_elementwise;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod index;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod logsumexp;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod math;
pub mod matmul;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod norm;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod reduce;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod scalar;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod softmax;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod softmax_bwd;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod special;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod unary;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub mod where_select;

// x86-64 only: streaming store utilities
#[cfg(target_arch = "x86_64")]
pub mod streaming;

pub use detect::{SimdLevel, detect_simd};
