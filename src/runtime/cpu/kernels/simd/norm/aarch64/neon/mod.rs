//! NEON normalization kernels for ARM64
//!
//! Provides vectorized RMS normalization and Layer normalization using 128-bit NEON registers.

pub(super) const F32_LANES: usize = 4;
pub(super) const F64_LANES: usize = 2;

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
