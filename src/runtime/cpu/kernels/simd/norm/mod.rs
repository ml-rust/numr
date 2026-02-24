//! SIMD-accelerated normalization operations
//!
//! This module provides AVX2 and AVX-512 implementations for normalization
//! operations critical in transformer architectures.
//!
//! # SIMD Optimizations
//!
//! - Sum of squares (for RMS): SIMD vertical accumulation + horizontal reduction
//! - Mean computation: SIMD sum + division
//! - Element-wise operations: SIMD multiply, add

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(feature = "f16")]
mod half;
#[cfg(feature = "f16")]
pub use half::{
    fused_add_layer_norm_bf16, fused_add_layer_norm_bwd_bf16, fused_add_layer_norm_bwd_f16,
    fused_add_layer_norm_f16, fused_add_rms_norm_bf16, fused_add_rms_norm_bwd_bf16,
    fused_add_rms_norm_bwd_f16, fused_add_rms_norm_f16, layer_norm_bf16, layer_norm_f16,
    rms_norm_bf16, rms_norm_f16,
};

/// Minimum hidden_size to justify SIMD overhead
pub(super) const SIMD_THRESHOLD: usize = 64;

mod fused_add_layer_norm;
mod fused_add_rms_norm;
mod layer_norm;
mod rms_norm;

pub use fused_add_layer_norm::{
    fused_add_layer_norm_bwd_f32, fused_add_layer_norm_bwd_f64, fused_add_layer_norm_f32,
    fused_add_layer_norm_f64,
};
pub use fused_add_rms_norm::{
    fused_add_rms_norm_bwd_f32, fused_add_rms_norm_bwd_f64, fused_add_rms_norm_f32,
    fused_add_rms_norm_f64,
};
pub use layer_norm::{layer_norm_f32, layer_norm_f64};
pub use rms_norm::{rms_norm_f32, rms_norm_f64};
