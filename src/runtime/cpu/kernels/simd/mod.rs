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
#[cfg(feature = "f16")]
pub mod half_convert_utils;

// Macros for generating f16/bf16 block-convert-compute wrappers (must come before users)
// Always compiled - macros internally gate generated code with #[cfg(feature = "f16")]
#[macro_use]
mod half_macros;

// CPU SIMD capability detection
mod detect;

// Operation modules - available on all architectures
// Each operation's mod.rs handles internal architecture dispatch
pub mod activations;
pub mod binary;
pub mod clamp;
pub mod compare;
pub mod conv;
pub mod cumulative;
pub mod dot;
pub mod fused_activation_mul;
pub mod fused_elementwise;
pub mod index;
pub mod logsumexp;
pub mod math;
pub mod matmul;
pub mod norm;
pub mod reduce;
pub mod scalar;
pub mod softmax;
pub mod softmax_bwd;
pub mod special;
pub mod unary;
pub mod where_select;

// x86-64 only: streaming store utilities
#[cfg(target_arch = "x86_64")]
pub mod streaming;

pub use detect::{SimdLevel, detect_simd};
