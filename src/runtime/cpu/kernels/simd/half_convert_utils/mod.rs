//! SIMD-accelerated f16/bf16 ↔ f32 conversion utilities
//!
//! These are the building blocks for the block-convert-compute pattern:
//! convert half-precision data to f32 in L1-sized blocks, run existing
//! f32 SIMD kernels, then convert back.
//!
//! # Conversion strategies
//!
//! - **x86 f16**: F16C instructions (`_mm256_cvtph_ps` / `_mm256_cvtps_ph`)
//! - **x86 bf16**: SIMD integer bit-shift (`u32 << 16` for load, rounded `>> 16` for store)
//! - **ARM f16**: NEON `vcvt_f32_f16` / `vcvt_f16_f32`
//! - **ARM bf16**: NEON integer bit-shift
//! - **Fallback**: `half` crate scalar conversion

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

mod dispatch;
mod scalar;

pub use dispatch::{
    HALF_BLOCK, convert_bf16_to_f32, convert_f16_to_f32, convert_f32_to_bf16, convert_f32_to_f16,
};
