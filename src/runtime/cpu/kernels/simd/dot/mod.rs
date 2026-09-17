//! SIMD-accelerated integer dot product operations
//!
//! Provides high-throughput i8 x i8 → i32 dot products for quantized inference.
//!
//! # Architecture Support
//!
//! | Architecture | Instruction Set  | Elements/cycle | Key Intrinsic          |
//! |--------------|------------------|----------------|------------------------|
//! | x86-64       | AVX-512BW        | 64             | maddubs + madd         |
//! | x86-64       | AVX2             | 32             | maddubs + madd         |
//! | ARM64        | NEON             | 16             | vmull_s8 + vpadalq_s16 |

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

mod dispatch;
mod scalar;

pub use dispatch::{i8xi8_dot_f32, i8xi8_dot_i32};
pub(in crate::runtime::cpu::kernels::simd) use scalar::{DOT_SPILL_ITERS, saturate_i64_to_i32};
