//! SIMD dispatch and scalar fallbacks for fused Add + RMS normalization

use super::super::{SimdLevel, detect_simd};
use super::SIMD_THRESHOLD;

#[cfg(target_arch = "x86_64")]
use super::avx2;
#[cfg(target_arch = "x86_64")]
use super::avx512;

#[cfg(target_arch = "aarch64")]
use super::aarch64;

// ============================================================================
// Fused Add + RMS Norm (forward)
// ============================================================================

/// SIMD Fused Add + RMS normalization for f32
#[inline]
pub unsafe fn fused_add_rms_norm_f32(
    input: *const f32,
    residual: *const f32,
    weight: *const f32,
    out: *mut f32,
    pre_norm: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let level = detect_simd();

    if hidden_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        fused_add_rms_norm_scalar_f32(
            input,
            residual,
            weight,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        );
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::fused_add_rms_norm_f32(
            input,
            residual,
            weight,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
        SimdLevel::Avx2Fma => avx2::fused_add_rms_norm_f32(
            input,
            residual,
            weight,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_rms_norm_scalar_f32(
            input,
            residual,
            weight,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::fused_add_rms_norm_f32(
            input,
            residual,
            weight,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_rms_norm_scalar_f32(
            input,
            residual,
            weight,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fused_add_rms_norm_scalar_f32(
        input,
        residual,
        weight,
        out,
        pre_norm,
        batch_size,
        hidden_size,
        eps,
    );
}

/// SIMD Fused Add + RMS normalization for f64
#[inline]
pub unsafe fn fused_add_rms_norm_f64(
    input: *const f64,
    residual: *const f64,
    weight: *const f64,
    out: *mut f64,
    pre_norm: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    let level = detect_simd();

    if hidden_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        fused_add_rms_norm_scalar_f64(
            input,
            residual,
            weight,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        );
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::fused_add_rms_norm_f64(
            input,
            residual,
            weight,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
        SimdLevel::Avx2Fma => avx2::fused_add_rms_norm_f64(
            input,
            residual,
            weight,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_rms_norm_scalar_f64(
            input,
            residual,
            weight,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::fused_add_rms_norm_f64(
            input,
            residual,
            weight,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_rms_norm_scalar_f64(
            input,
            residual,
            weight,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fused_add_rms_norm_scalar_f64(
        input,
        residual,
        weight,
        out,
        pre_norm,
        batch_size,
        hidden_size,
        eps,
    );
}

// ============================================================================
// Fused Add + RMS Norm (backward)
// ============================================================================

/// SIMD Fused Add + RMS Norm Backward for f32
#[inline]
pub unsafe fn fused_add_rms_norm_bwd_f32(
    grad: *const f32,
    pre_norm: *const f32,
    weight: *const f32,
    d_input_residual: *mut f32,
    d_weight: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let level = detect_simd();

    if hidden_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        fused_add_rms_norm_bwd_scalar_f32(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            batch_size,
            hidden_size,
            eps,
        );
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::fused_add_rms_norm_bwd_f32(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            batch_size,
            hidden_size,
            eps,
        ),
        SimdLevel::Avx2Fma => avx2::fused_add_rms_norm_bwd_f32(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_rms_norm_bwd_scalar_f32(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::fused_add_rms_norm_bwd_f32(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_rms_norm_bwd_scalar_f32(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fused_add_rms_norm_bwd_scalar_f32(
        grad,
        pre_norm,
        weight,
        d_input_residual,
        d_weight,
        batch_size,
        hidden_size,
        eps,
    );
}

/// SIMD Fused Add + RMS Norm Backward for f64
#[inline]
pub unsafe fn fused_add_rms_norm_bwd_f64(
    grad: *const f64,
    pre_norm: *const f64,
    weight: *const f64,
    d_input_residual: *mut f64,
    d_weight: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    let level = detect_simd();

    if hidden_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        fused_add_rms_norm_bwd_scalar_f64(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            batch_size,
            hidden_size,
            eps,
        );
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::fused_add_rms_norm_bwd_f64(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            batch_size,
            hidden_size,
            eps,
        ),
        SimdLevel::Avx2Fma => avx2::fused_add_rms_norm_bwd_f64(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_rms_norm_bwd_scalar_f64(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::fused_add_rms_norm_bwd_f64(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_rms_norm_bwd_scalar_f64(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fused_add_rms_norm_bwd_scalar_f64(
        grad,
        pre_norm,
        weight,
        d_input_residual,
        d_weight,
        batch_size,
        hidden_size,
        eps,
    );
}

// ============================================================================
// Scalar fallbacks for fused add + RMS norm
// ============================================================================

/// Scalar fused add + RMS norm for f32
#[inline]
pub unsafe fn fused_add_rms_norm_scalar_f32(
    input: *const f32,
    residual: *const f32,
    weight: *const f32,
    out: *mut f32,
    pre_norm: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // Add and store pre_norm, compute sum of squares
        let mut sum_sq = 0.0f32;
        for i in 0..hidden_size {
            let pn = *input.add(row_start + i) + *residual.add(row_start + i);
            *pre_norm.add(row_start + i) = pn;
            sum_sq += pn * pn;
        }

        let inv_rms = 1.0 / (sum_sq / hidden_size as f32 + eps).sqrt();

        for i in 0..hidden_size {
            let pn = *pre_norm.add(row_start + i);
            let w = *weight.add(i);
            *out.add(row_start + i) = pn * inv_rms * w;
        }
    }
}

/// Scalar fused add + RMS norm for f64
#[inline]
pub unsafe fn fused_add_rms_norm_scalar_f64(
    input: *const f64,
    residual: *const f64,
    weight: *const f64,
    out: *mut f64,
    pre_norm: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        let mut sum_sq = 0.0f64;
        for i in 0..hidden_size {
            let pn = *input.add(row_start + i) + *residual.add(row_start + i);
            *pre_norm.add(row_start + i) = pn;
            sum_sq += pn * pn;
        }

        let inv_rms = 1.0 / (sum_sq / hidden_size as f64 + eps).sqrt();

        for i in 0..hidden_size {
            let pn = *pre_norm.add(row_start + i);
            let w = *weight.add(i);
            *out.add(row_start + i) = pn * inv_rms * w;
        }
    }
}

/// Scalar fused add + RMS norm backward for f32
#[inline]
pub unsafe fn fused_add_rms_norm_bwd_scalar_f32(
    grad: *const f32,
    pre_norm: *const f32,
    weight: *const f32,
    d_input_residual: *mut f32,
    d_weight: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        let mut sum_sq = 0.0f32;
        for i in 0..hidden_size {
            let pn = *pre_norm.add(row_start + i);
            sum_sq += pn * pn;
        }

        let mean_sq = sum_sq / hidden_size as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        let mut dot = 0.0f32;
        for i in 0..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i);
            dot += g * w * pn;
        }

        let coeff = dot * inv_rms / (hidden_size as f32 * (mean_sq + eps));

        for i in 0..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i);

            let d_ir = (g * w - pn * coeff) * inv_rms;
            *d_input_residual.add(row_start + i) = d_ir;

            let d_w = g * pn * inv_rms;
            *d_weight.add(i) += d_w;
        }
    }
}

/// Scalar fused add + RMS norm backward for f64
#[inline]
pub unsafe fn fused_add_rms_norm_bwd_scalar_f64(
    grad: *const f64,
    pre_norm: *const f64,
    weight: *const f64,
    d_input_residual: *mut f64,
    d_weight: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        let mut sum_sq = 0.0f64;
        for i in 0..hidden_size {
            let pn = *pre_norm.add(row_start + i);
            sum_sq += pn * pn;
        }

        let mean_sq = sum_sq / hidden_size as f64;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();

        let mut dot = 0.0f64;
        for i in 0..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i);
            dot += g * w * pn;
        }

        let coeff = dot * inv_rms / (hidden_size as f64 * (mean_sq + eps));

        for i in 0..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i);

            let d_ir = (g * w - pn * coeff) * inv_rms;
            *d_input_residual.add(row_start + i) = d_ir;

            let d_w = g * pn * inv_rms;
            *d_weight.add(i) += d_w;
        }
    }
}
