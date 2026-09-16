//! Stockham autosort radix-2 FFT kernels (power-of-two sizes)

use crate::dtype::{Complex64, Complex128};
use std::f64::consts::PI;

use super::bluestein::BluesteinPlan;
#[cfg(feature = "rayon")]
use super::{fft_c64, fft_c128};

// ============================================================================
// Complex64 (f32) FFT Kernels
// ============================================================================

/// Stockham FFT for Complex64 data
///
/// # Arguments
///
/// * `input` - Input complex data slice of length N (must be power of 2)
/// * `output` - Output buffer, will be filled with FFT result
/// * `inverse` - If true, compute inverse FFT
/// * `normalize_factor` - Scale factor to apply to each output element
///
/// # Safety
///
/// * `input` and `output` must be valid slices of length N
/// * N must be a power of 2
pub(super) unsafe fn stockham_fft_c64(
    input: &[Complex64],
    output: &mut [Complex64],
    inverse: bool,
    normalize_factor: f32,
) {
    let n = input.len();
    debug_assert!(n > 0 && (n & (n - 1)) == 0, "N must be power of 2");
    debug_assert_eq!(input.len(), output.len());

    if n == 1 {
        output[0] = Complex64::new(
            input[0].re * normalize_factor,
            input[0].im * normalize_factor,
        );
        return;
    }

    let log_n = n.trailing_zeros() as usize;
    let sign = if inverse { 1.0f64 } else { -1.0f64 };

    // Double buffering - allocate working buffers
    let mut buf_a: Vec<Complex64> = input.to_vec();
    let mut buf_b: Vec<Complex64> = vec![Complex64::default(); n];

    // Reference to current source and destination
    let mut src = &mut buf_a;
    let mut dst = &mut buf_b;

    // Process each stage
    for stage in 0..log_n {
        let m = 1 << (stage + 1); // 2, 4, 8, ..., N
        let half_m = 1 << stage; // 1, 2, 4, ..., N/2
        let groups = n / m;

        // Process all butterflies in this stage
        for g in 0..groups {
            for b in 0..half_m {
                // Twiddle factor: W_m^b = exp(sign * 2πi * b / m)
                let theta = sign * 2.0 * PI * (b as f64) / (m as f64);
                let twiddle = Complex64::new(theta.cos() as f32, theta.sin() as f32);

                // Stockham addressing:
                // Even elements: src[g * half_m + b]
                // Odd elements:  src[N/2 + g * half_m + b]
                let even_idx = g * half_m + b;
                let odd_idx = n / 2 + g * half_m + b;

                let even = src[even_idx];
                let odd = src[odd_idx] * twiddle;

                // Output addresses for this stage
                let out_idx_lo = g * m + b;
                let out_idx_hi = g * m + b + half_m;

                dst[out_idx_lo] = even + odd;
                dst[out_idx_hi] = even - odd;
            }
        }

        // Swap buffers for next stage
        std::mem::swap(&mut src, &mut dst);
    }

    // Result is in src after final swap
    // Apply normalization factor and copy to output
    for i in 0..n {
        output[i] = Complex64::new(src[i].re * normalize_factor, src[i].im * normalize_factor);
    }
}

/// Batched FFT for Complex64 data
///
/// Processes multiple independent FFTs in parallel. Power-of-two sizes use the
/// Stockham kernel; any other size uses Bluestein's algorithm with a single plan
/// shared across the batch.
///
/// # Safety
///
/// * `input` and `output` must have length `batch_size * n`
/// * n must be >= 1
#[cfg(feature = "rayon")]
pub unsafe fn stockham_fft_batched_c64(
    input: &[Complex64],
    output: &mut [Complex64],
    n: usize,
    batch_size: usize,
    inverse: bool,
    normalize_factor: f32,
    min_batch_len: usize,
) {
    use rayon::prelude::*;

    debug_assert_eq!(input.len(), batch_size * n);
    debug_assert_eq!(output.len(), batch_size * n);

    // Single-batch: call directly to avoid Rayon thread pool overhead (~15-20%)
    if batch_size == 1 {
        fft_c64(input, output, inverse, normalize_factor);
        return;
    }

    if n.is_power_of_two() {
        output
            .par_chunks_mut(n)
            .enumerate()
            .with_min_len(min_batch_len.max(1))
            .for_each(|(batch_idx, out_chunk)| {
                let in_start = batch_idx * n;
                let in_chunk = &input[in_start..in_start + n];
                stockham_fft_c64(in_chunk, out_chunk, inverse, normalize_factor);
            });
        return;
    }

    let plan = BluesteinPlan::new(n, inverse);
    output
        .par_chunks_mut(n)
        .enumerate()
        .with_min_len(min_batch_len.max(1))
        .for_each(|(batch_idx, out_chunk)| {
            let in_start = batch_idx * n;
            let in_chunk = &input[in_start..in_start + n];
            plan.execute_c64(in_chunk, out_chunk, normalize_factor);
        });
}

#[cfg(not(feature = "rayon"))]
pub unsafe fn stockham_fft_batched_c64(
    input: &[Complex64],
    output: &mut [Complex64],
    n: usize,
    batch_size: usize,
    inverse: bool,
    normalize_factor: f32,
    _min_batch_len: usize,
) {
    debug_assert_eq!(input.len(), batch_size * n);
    debug_assert_eq!(output.len(), batch_size * n);

    if n.is_power_of_two() {
        for batch_idx in 0..batch_size {
            let start = batch_idx * n;
            let end = start + n;
            stockham_fft_c64(
                &input[start..end],
                &mut output[start..end],
                inverse,
                normalize_factor,
            );
        }
        return;
    }

    let plan = BluesteinPlan::new(n, inverse);
    for batch_idx in 0..batch_size {
        let start = batch_idx * n;
        let end = start + n;
        plan.execute_c64(
            &input[start..end],
            &mut output[start..end],
            normalize_factor,
        );
    }
}

// ============================================================================
// Complex128 (f64) FFT Kernels
// ============================================================================

/// Stockham FFT for Complex128 data
///
/// # Arguments
///
/// * `input` - Input complex data slice of length N (must be power of 2)
/// * `output` - Output buffer, will be filled with FFT result
/// * `inverse` - If true, compute inverse FFT
/// * `normalize_factor` - Scale factor to apply to each output element
///
/// # Safety
///
/// * `input` and `output` must be valid slices of length N
/// * N must be a power of 2
pub(super) unsafe fn stockham_fft_c128(
    input: &[Complex128],
    output: &mut [Complex128],
    inverse: bool,
    normalize_factor: f64,
) {
    let n = input.len();
    debug_assert!(n > 0 && (n & (n - 1)) == 0, "N must be power of 2");
    debug_assert_eq!(input.len(), output.len());

    if n == 1 {
        output[0] = Complex128::new(
            input[0].re * normalize_factor,
            input[0].im * normalize_factor,
        );
        return;
    }

    let log_n = n.trailing_zeros() as usize;
    let sign = if inverse { 1.0f64 } else { -1.0f64 };

    // Double buffering
    let mut buf_a: Vec<Complex128> = input.to_vec();
    let mut buf_b: Vec<Complex128> = vec![Complex128::default(); n];

    let mut src = &mut buf_a;
    let mut dst = &mut buf_b;

    for stage in 0..log_n {
        let m = 1 << (stage + 1);
        let half_m = 1 << stage;
        let groups = n / m;

        for g in 0..groups {
            for b in 0..half_m {
                let theta = sign * 2.0 * PI * (b as f64) / (m as f64);
                let twiddle = Complex128::new(theta.cos(), theta.sin());

                let even_idx = g * half_m + b;
                let odd_idx = n / 2 + g * half_m + b;

                let even = src[even_idx];
                let odd = src[odd_idx] * twiddle;

                let out_idx_lo = g * m + b;
                let out_idx_hi = g * m + b + half_m;

                dst[out_idx_lo] = even + odd;
                dst[out_idx_hi] = even - odd;
            }
        }

        std::mem::swap(&mut src, &mut dst);
    }

    for i in 0..n {
        output[i] = Complex128::new(src[i].re * normalize_factor, src[i].im * normalize_factor);
    }
}

/// Batched FFT for Complex128 data
///
/// Power-of-two sizes use the Stockham kernel; any other size uses Bluestein's
/// algorithm with a single plan shared across the batch.
///
/// # Safety
///
/// * `input` and `output` must have length `batch_size * n`
/// * n must be >= 1
#[cfg(feature = "rayon")]
pub unsafe fn stockham_fft_batched_c128(
    input: &[Complex128],
    output: &mut [Complex128],
    n: usize,
    batch_size: usize,
    inverse: bool,
    normalize_factor: f64,
    min_batch_len: usize,
) {
    use rayon::prelude::*;

    debug_assert_eq!(input.len(), batch_size * n);
    debug_assert_eq!(output.len(), batch_size * n);

    // Single-batch: call directly to avoid Rayon thread pool overhead (~15-20%)
    if batch_size == 1 {
        fft_c128(input, output, inverse, normalize_factor);
        return;
    }

    if n.is_power_of_two() {
        output
            .par_chunks_mut(n)
            .enumerate()
            .with_min_len(min_batch_len.max(1))
            .for_each(|(batch_idx, out_chunk)| {
                let in_start = batch_idx * n;
                let in_chunk = &input[in_start..in_start + n];
                stockham_fft_c128(in_chunk, out_chunk, inverse, normalize_factor);
            });
        return;
    }

    let plan = BluesteinPlan::new(n, inverse);
    output
        .par_chunks_mut(n)
        .enumerate()
        .with_min_len(min_batch_len.max(1))
        .for_each(|(batch_idx, out_chunk)| {
            let in_start = batch_idx * n;
            let in_chunk = &input[in_start..in_start + n];
            plan.execute_c128(in_chunk, out_chunk, normalize_factor);
        });
}

#[cfg(not(feature = "rayon"))]
pub unsafe fn stockham_fft_batched_c128(
    input: &[Complex128],
    output: &mut [Complex128],
    n: usize,
    batch_size: usize,
    inverse: bool,
    normalize_factor: f64,
    _min_batch_len: usize,
) {
    debug_assert_eq!(input.len(), batch_size * n);
    debug_assert_eq!(output.len(), batch_size * n);

    if n.is_power_of_two() {
        for batch_idx in 0..batch_size {
            let start = batch_idx * n;
            let end = start + n;
            stockham_fft_c128(
                &input[start..end],
                &mut output[start..end],
                inverse,
                normalize_factor,
            );
        }
        return;
    }

    let plan = BluesteinPlan::new(n, inverse);
    for batch_idx in 0..batch_size {
        let start = batch_idx * n;
        let end = start + n;
        plan.execute_c128(
            &input[start..end],
            &mut output[start..end],
            normalize_factor,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_impulse() {
        // FFT of [1, 0, 0, 0] should be [1, 1, 1, 1]
        let input = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let mut output = [Complex64::default(); 4];

        unsafe {
            stockham_fft_c64(&input, &mut output, false, 1.0);
        }

        for c in &output {
            assert!((c.re - 1.0).abs() < 1e-5, "Expected 1.0, got {}", c.re);
            assert!(c.im.abs() < 1e-5, "Expected 0.0i, got {}i", c.im);
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip() {
        // FFT followed by IFFT should recover original signal
        let input = [
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(7.0, 8.0),
        ];
        let mut fft_output = [Complex64::default(); 4];
        let mut ifft_output = [Complex64::default(); 4];

        unsafe {
            // Forward FFT (no normalization)
            stockham_fft_c64(&input, &mut fft_output, false, 1.0);
            // Inverse FFT (normalize by 1/N = 0.25)
            stockham_fft_c64(&fft_output, &mut ifft_output, true, 0.25);
        }

        for i in 0..4 {
            assert!(
                (ifft_output[i].re - input[i].re).abs() < 1e-5,
                "Real mismatch at {}: {} vs {}",
                i,
                ifft_output[i].re,
                input[i].re
            );
            assert!(
                (ifft_output[i].im - input[i].im).abs() < 1e-5,
                "Imag mismatch at {}: {} vs {}",
                i,
                ifft_output[i].im,
                input[i].im
            );
        }
    }

    #[test]
    fn test_fft_parseval() {
        // Parseval's theorem: sum(|x|^2) = (1/N) * sum(|X|^2)
        let input = [
            Complex64::new(1.0, 0.5),
            Complex64::new(2.0, 1.0),
            Complex64::new(0.5, 0.5),
            Complex64::new(1.5, 0.0),
        ];
        let mut output = [Complex64::default(); 4];

        unsafe {
            stockham_fft_c64(&input, &mut output, false, 1.0);
        }

        let energy_time: f32 = input.iter().map(|c| c.re * c.re + c.im * c.im).sum();
        let energy_freq: f32 = output.iter().map(|c| c.re * c.re + c.im * c.im).sum();

        // energy_time = (1/N) * energy_freq
        let expected_freq_energy = energy_time * 4.0;
        assert!(
            (energy_freq - expected_freq_energy).abs() < 1e-4,
            "Parseval failed: {} vs {}",
            energy_freq,
            expected_freq_energy
        );
    }

    #[test]
    fn test_fft_size_2() {
        // Simple N=2 case
        let input = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let mut output = [Complex64::default(); 2];

        unsafe {
            stockham_fft_c64(&input, &mut output, false, 1.0);
        }

        // X[0] = x[0] + x[1] = 3
        // X[1] = x[0] - x[1] = -1
        assert!((output[0].re - 3.0).abs() < 1e-5);
        assert!(output[0].im.abs() < 1e-5);
        assert!((output[1].re - (-1.0)).abs() < 1e-5);
        assert!(output[1].im.abs() < 1e-5);
    }

    #[test]
    fn test_fft_c128() {
        // Test f64 precision FFT
        let input = [
            Complex128::new(1.0, 0.0),
            Complex128::new(0.0, 0.0),
            Complex128::new(0.0, 0.0),
            Complex128::new(0.0, 0.0),
        ];
        let mut output = [Complex128::default(); 4];

        unsafe {
            stockham_fft_c128(&input, &mut output, false, 1.0);
        }

        for c in &output {
            assert!((c.re - 1.0).abs() < 1e-10);
            assert!(c.im.abs() < 1e-10);
        }
    }
}
