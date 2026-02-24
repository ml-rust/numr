//! AVX2 normalization kernels
//!
//! SIMD-optimized RMS norm and layer norm with manual horizontal reductions.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(super) const F32_LANES: usize = 8;
pub(super) const F64_LANES: usize = 4;

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

// ============================================================================
// Horizontal reduction helpers (used by sub-modules)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub(super) unsafe fn hsum_f32(v: __m256) -> f32 {
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(low, high);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub(super) unsafe fn hsum_f64(v: __m256d) -> f64 {
    let high = _mm256_extractf128_pd(v, 1);
    let low = _mm256_castpd256_pd128(v);
    let sum128 = _mm_add_pd(low, high);
    let shuf = _mm_unpackhi_pd(sum128, sum128);
    let sum64 = _mm_add_sd(sum128, shuf);
    _mm_cvtsd_f64(sum64)
}
