//! SIMD dispatch and scalar fallbacks for RMS normalization

use super::super::{SimdLevel, detect_simd};
use super::SIMD_THRESHOLD;

#[cfg(target_arch = "x86_64")]
use super::avx2;
#[cfg(target_arch = "x86_64")]
use super::avx512;

#[cfg(target_arch = "aarch64")]
use super::aarch64;

/// SIMD RMS normalization for f32
///
/// # Safety
/// - `input` and `out` must point to `batch_size * hidden_size` elements
/// - `weight` must point to `hidden_size` elements
#[inline]
pub unsafe fn rms_norm_f32(
    input: *const f32,
    weight: *const f32,
    out: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let level = detect_simd();

    if hidden_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        rms_norm_scalar_f32(input, weight, out, batch_size, hidden_size, eps);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::rms_norm_f32(input, weight, out, batch_size, hidden_size, eps),
        SimdLevel::Avx2Fma => avx2::rms_norm_f32(input, weight, out, batch_size, hidden_size, eps),
        _ => rms_norm_scalar_f32(input, weight, out, batch_size, hidden_size, eps),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::rms_norm_f32(input, weight, out, batch_size, hidden_size, eps)
        }
        _ => rms_norm_scalar_f32(input, weight, out, batch_size, hidden_size, eps),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    rms_norm_scalar_f32(input, weight, out, batch_size, hidden_size, eps);
}

/// SIMD RMS normalization for f64
///
/// # Safety
/// - `input` and `out` must point to `batch_size * hidden_size` elements
/// - `weight` must point to `hidden_size` elements
#[inline]
pub unsafe fn rms_norm_f64(
    input: *const f64,
    weight: *const f64,
    out: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    let level = detect_simd();

    if hidden_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        rms_norm_scalar_f64(input, weight, out, batch_size, hidden_size, eps);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::rms_norm_f64(input, weight, out, batch_size, hidden_size, eps),
        SimdLevel::Avx2Fma => avx2::rms_norm_f64(input, weight, out, batch_size, hidden_size, eps),
        _ => rms_norm_scalar_f64(input, weight, out, batch_size, hidden_size, eps),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::rms_norm_f64(input, weight, out, batch_size, hidden_size, eps)
        }
        _ => rms_norm_scalar_f64(input, weight, out, batch_size, hidden_size, eps),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    rms_norm_scalar_f64(input, weight, out, batch_size, hidden_size, eps);
}

/// Scalar RMS norm for f32
#[inline]
pub unsafe fn rms_norm_scalar_f32(
    input: *const f32,
    weight: *const f32,
    out: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // Compute sum of squares in f64 for precision (matches llama.cpp's ggml_float)
        let mut sum_sq = 0.0f64;
        for i in 0..hidden_size {
            let x = *input.add(row_start + i) as f64;
            sum_sq += x * x;
        }

        // Compute inverse RMS in f64, then cast to f32
        let inv_rms = (1.0f64 / (sum_sq / hidden_size as f64 + eps as f64).sqrt()) as f32;

        // Apply normalization and weight
        for i in 0..hidden_size {
            let x = *input.add(row_start + i);
            let w = *weight.add(i);
            *out.add(row_start + i) = x * inv_rms * w;
        }
    }
}

/// Scalar RMS norm for f64
#[inline]
pub unsafe fn rms_norm_scalar_f64(
    input: *const f64,
    weight: *const f64,
    out: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        let mut sum_sq = 0.0f64;
        for i in 0..hidden_size {
            let x = *input.add(row_start + i);
            sum_sq += x * x;
        }

        let inv_rms = 1.0 / (sum_sq / hidden_size as f64 + eps).sqrt();

        for i in 0..hidden_size {
            let x = *input.add(row_start + i);
            let w = *weight.add(i);
            *out.add(row_start + i) = x * inv_rms * w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_f32() {
        let hidden_size = 128;
        let batch_size = 4;
        let input: Vec<f32> = (0..(batch_size * hidden_size))
            .map(|x| (x as f32) / 100.0 - 2.5)
            .collect();
        let weight: Vec<f32> = vec![1.0f32; hidden_size];
        let mut out = vec![0.0f32; batch_size * hidden_size];
        let mut out_ref = vec![0.0f32; batch_size * hidden_size];

        unsafe {
            rms_norm_f32(
                input.as_ptr(),
                weight.as_ptr(),
                out.as_mut_ptr(),
                batch_size,
                hidden_size,
                1e-5,
            );
            rms_norm_scalar_f32(
                input.as_ptr(),
                weight.as_ptr(),
                out_ref.as_mut_ptr(),
                batch_size,
                hidden_size,
                1e-5,
            );
        }

        for i in 0..(batch_size * hidden_size) {
            assert!(
                (out[i] - out_ref[i]).abs() < 1e-4,
                "mismatch at {}: {} vs {}",
                i,
                out[i],
                out_ref[i]
            );
        }
    }
}
