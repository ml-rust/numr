//! SIMD-accelerated unary operations
//!
//! This module provides multi-architecture SIMD implementations for element-wise
//! unary operations.
//!
//! # SIMD Support
//!
//! ALL operations now have SIMD implementations:
//! - Neg, Abs, Sqrt, Square, Recip, Floor, Ceil, Round, RoundTiesEven, Trunc
//!   (direct SIMD)
//! - Exp, Log, Sin, Cos, Tan, Atan, Tanh (polynomial approximations from math module)
//! - Sign (comparison-based)
//! - ReLU (critical for ML)
//!
//! # Architecture Support
//!
//! | Architecture | Instruction Set | Vector Width | f32 lanes | f64 lanes |
//! |--------------|-----------------|--------------|-----------|-----------|
//! | x86-64       | AVX-512         | 512 bits     | 16        | 8         |
//! | x86-64       | AVX2 + FMA      | 256 bits     | 8         | 4         |
//! | ARM64        | NEON            | 128 bits     | 4         | 2         |

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

mod dispatch;

#[cfg(test)]
mod test_support;

mod basic_ops;
mod exp_f32;
mod exp_f64;
mod hyperbolic_f32;
mod hyperbolic_f64;
mod inverse_trig;
mod log_f32;
mod log_f64;
mod trig_f32;
mod trig_f64;

// Import scalar fallbacks from kernels module (single source of truth)
pub use crate::runtime::cpu::kernels::unary::{
    relu_scalar_f32, relu_scalar_f64, unary_scalar_f32, unary_scalar_f64,
};

pub use dispatch::*;
