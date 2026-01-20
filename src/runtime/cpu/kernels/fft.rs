//! FFT kernels using Stockham autosort algorithm
//!
//! This module provides CPU implementations of FFT operations.
//! The Stockham algorithm is used for its:
//! - No bit-reversal permutation (Cooley-Tukey's main bottleneck)
//! - Sequential memory access patterns
//! - Natural double-buffering
//!
//! # Algorithm: Stockham Radix-2 FFT
//!
//! ```text
//! For each stage s = 0..log2(N):
//!     half_m = 2^s
//!     m = 2^(s+1)
//!     For each group g = 0..(N/m):
//!         For each butterfly b = 0..half_m:
//!             twiddle = exp(sign * 2πi * b / m)
//!             even = src[g * half_m + b]
//!             odd = src[N/2 + g * half_m + b] * twiddle
//!             dst[g * m + b] = even + odd
//!             dst[g * m + b + half_m] = even - odd
//!     swap(src, dst)
//! ```

use crate::dtype::{Complex64, Complex128};
use std::f64::consts::PI;

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
pub unsafe fn stockham_fft_c64(
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

/// Batched Stockham FFT for Complex64 data
///
/// Processes multiple independent FFTs in parallel.
///
/// # Safety
///
/// * `input` and `output` must have length `batch_size * n`
/// * n must be a power of 2
#[cfg(feature = "rayon")]
pub unsafe fn stockham_fft_batched_c64(
    input: &[Complex64],
    output: &mut [Complex64],
    n: usize,
    batch_size: usize,
    inverse: bool,
    normalize_factor: f32,
) {
    use rayon::prelude::*;

    debug_assert_eq!(input.len(), batch_size * n);
    debug_assert_eq!(output.len(), batch_size * n);

    let output_chunks = std::slice::from_raw_parts_mut(output.as_mut_ptr(), batch_size * n);

    // Process batches in parallel
    output_chunks
        .chunks_mut(n)
        .enumerate()
        .par_bridge()
        .for_each(|(batch_idx, out_chunk)| {
            let in_start = batch_idx * n;
            let in_chunk = &input[in_start..in_start + n];
            stockham_fft_c64(in_chunk, out_chunk, inverse, normalize_factor);
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
) {
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
pub unsafe fn stockham_fft_c128(
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

/// Batched Stockham FFT for Complex128 data
#[cfg(feature = "rayon")]
pub unsafe fn stockham_fft_batched_c128(
    input: &[Complex128],
    output: &mut [Complex128],
    n: usize,
    batch_size: usize,
    inverse: bool,
    normalize_factor: f64,
) {
    use rayon::prelude::*;

    debug_assert_eq!(input.len(), batch_size * n);
    debug_assert_eq!(output.len(), batch_size * n);

    let output_chunks = std::slice::from_raw_parts_mut(output.as_mut_ptr(), batch_size * n);

    output_chunks
        .chunks_mut(n)
        .enumerate()
        .par_bridge()
        .for_each(|(batch_idx, out_chunk)| {
            let in_start = batch_idx * n;
            let in_chunk = &input[in_start..in_start + n];
            stockham_fft_c128(in_chunk, out_chunk, inverse, normalize_factor);
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
) {
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
}

// ============================================================================
// Real FFT Kernels (rfft/irfft)
// ============================================================================

/// Real-to-complex FFT using the "packing" trick
///
/// For N real inputs, we pack them as N/2 complex values:
///   z[k] = x[2k] + i*x[2k+1]
///
/// Then compute N/2-point complex FFT and unpack to get N/2+1 complex outputs.
///
/// # Arguments
///
/// * `input` - Real input data of length N (must be power of 2)
/// * `output` - Complex output buffer of length N/2 + 1
///
/// # Safety
///
/// * N must be a power of 2
/// * `output` must have length N/2 + 1
pub unsafe fn rfft_c64(input: &[f32], output: &mut [Complex64], normalize_factor: f32) {
    let n = input.len();
    debug_assert!(n >= 2 && (n & (n - 1)) == 0, "N must be power of 2 >= 2");
    debug_assert_eq!(output.len(), n / 2 + 1);

    let half_n = n / 2;

    // Step 1: Pack real values into complex
    let mut packed: Vec<Complex64> = Vec::with_capacity(half_n);
    for k in 0..half_n {
        packed.push(Complex64::new(input[2 * k], input[2 * k + 1]));
    }

    // Step 2: Compute half-size complex FFT (no normalization yet)
    let mut fft_result = vec![Complex64::default(); half_n];
    stockham_fft_c64(&packed, &mut fft_result, false, 1.0);

    // Step 3: Unpack to get full rfft output
    // X[0] = Z[0].re + Z[0].im (DC component)
    // X[N/2] = Z[0].re - Z[0].im (Nyquist component)
    // X[k] = (Z[k] + conj(Z[N/2-k])) / 2 - i * (Z[k] - conj(Z[N/2-k])) / 2 * W_N^k
    //
    // Simplified unpack formula:
    // Xe[k] = (Z[k] + conj(Z[N/2-k])) / 2
    // Xo[k] = (Z[k] - conj(Z[N/2-k])) / 2i
    // X[k] = Xe[k] + W_N^(-k) * Xo[k]

    // DC component (k=0)
    output[0] = Complex64::new(
        (fft_result[0].re + fft_result[0].im) * normalize_factor,
        0.0,
    );

    // Middle components (k = 1 to N/2 - 1)
    for k in 1..half_n {
        let z_k = fft_result[k];
        let z_nk = fft_result[half_n - k].conj();

        let x_even = (z_k + z_nk) * Complex64::new(0.5, 0.0);
        let x_odd = (z_k - z_nk) * Complex64::new(0.0, -0.5);

        // Twiddle factor W_N^(-k)
        let theta = -2.0 * PI * (k as f64) / (n as f64);
        let twiddle = Complex64::new(theta.cos() as f32, theta.sin() as f32);

        let result = x_even + x_odd * twiddle;
        output[k] = Complex64::new(result.re * normalize_factor, result.im * normalize_factor);
    }

    // Nyquist component (k = N/2)
    output[half_n] = Complex64::new(
        (fft_result[0].re - fft_result[0].im) * normalize_factor,
        0.0,
    );
}

/// Complex-to-real inverse FFT
///
/// Takes Hermitian-symmetric complex input (N/2+1 values) and produces N real values.
///
/// # Safety
///
/// * `input` must have length N/2 + 1
/// * `output` must have length N
/// * N must be power of 2
pub unsafe fn irfft_c64(input: &[Complex64], output: &mut [f32], normalize_factor: f32) {
    let half_n_plus_1 = input.len();
    debug_assert!(half_n_plus_1 >= 2);
    let half_n = half_n_plus_1 - 1;
    let n = half_n * 2;
    debug_assert_eq!(output.len(), n);

    // Step 1: Extend Hermitian-symmetric input to full complex spectrum
    let mut full_spectrum = vec![Complex64::default(); n];
    full_spectrum[0] = input[0];
    for k in 1..half_n {
        full_spectrum[k] = input[k];
        full_spectrum[n - k] = input[k].conj();
    }
    full_spectrum[half_n] = input[half_n];

    // Step 2: Compute inverse FFT (with normalization 1/N built-in)
    let mut ifft_result = vec![Complex64::default(); n];
    stockham_fft_c64(&full_spectrum, &mut ifft_result, true, normalize_factor);

    // Step 3: Extract real parts
    for i in 0..n {
        output[i] = ifft_result[i].re;
    }
}

/// Real-to-complex FFT (f64 precision)
pub unsafe fn rfft_c128(input: &[f64], output: &mut [Complex128], normalize_factor: f64) {
    let n = input.len();
    debug_assert!(n >= 2 && (n & (n - 1)) == 0, "N must be power of 2 >= 2");
    debug_assert_eq!(output.len(), n / 2 + 1);

    let half_n = n / 2;

    // Pack real values into complex
    let mut packed: Vec<Complex128> = Vec::with_capacity(half_n);
    for k in 0..half_n {
        packed.push(Complex128::new(input[2 * k], input[2 * k + 1]));
    }

    // Compute half-size complex FFT
    let mut fft_result = vec![Complex128::default(); half_n];
    stockham_fft_c128(&packed, &mut fft_result, false, 1.0);

    // Unpack to get full rfft output
    output[0] = Complex128::new(
        (fft_result[0].re + fft_result[0].im) * normalize_factor,
        0.0,
    );

    for k in 1..half_n {
        let z_k = fft_result[k];
        let z_nk = fft_result[half_n - k].conj();

        let x_even = (z_k + z_nk) * Complex128::new(0.5, 0.0);
        let x_odd = (z_k - z_nk) * Complex128::new(0.0, -0.5);

        let theta = -2.0 * PI * (k as f64) / (n as f64);
        let twiddle = Complex128::new(theta.cos(), theta.sin());

        let result = x_even + x_odd * twiddle;
        output[k] = Complex128::new(result.re * normalize_factor, result.im * normalize_factor);
    }

    output[half_n] = Complex128::new(
        (fft_result[0].re - fft_result[0].im) * normalize_factor,
        0.0,
    );
}

/// Complex-to-real inverse FFT (f64 precision)
pub unsafe fn irfft_c128(input: &[Complex128], output: &mut [f64], normalize_factor: f64) {
    let half_n_plus_1 = input.len();
    debug_assert!(half_n_plus_1 >= 2);
    let half_n = half_n_plus_1 - 1;
    let n = half_n * 2;
    debug_assert_eq!(output.len(), n);

    // Extend Hermitian-symmetric input
    let mut full_spectrum = vec![Complex128::default(); n];
    full_spectrum[0] = input[0];
    for k in 1..half_n {
        full_spectrum[k] = input[k];
        full_spectrum[n - k] = input[k].conj();
    }
    full_spectrum[half_n] = input[half_n];

    // Compute inverse FFT
    let mut ifft_result = vec![Complex128::default(); n];
    stockham_fft_c128(&full_spectrum, &mut ifft_result, true, normalize_factor);

    // Extract real parts
    for i in 0..n {
        output[i] = ifft_result[i].re;
    }
}

// ============================================================================
// FFT Shift Kernels
// ============================================================================

/// Shift zero-frequency component to center
///
/// For 1D array of length N, swaps [0..N/2] with [N/2..N]
pub unsafe fn fftshift_c64(input: &[Complex64], output: &mut [Complex64]) {
    let n = input.len();
    let half_n = n / 2;

    // Copy second half to first half of output
    for i in 0..half_n {
        output[i] = input[half_n + i];
    }
    // Copy first half to second half of output
    for i in 0..n - half_n {
        output[half_n + i] = input[i];
    }
}

/// Inverse shift (undo fftshift)
pub unsafe fn ifftshift_c64(input: &[Complex64], output: &mut [Complex64]) {
    let n = input.len();
    let half_n = (n + 1) / 2; // For odd lengths, first half is larger

    // For ifftshift: swap [0..ceil(N/2)] with [ceil(N/2)..N]
    let shift = n - half_n;
    for i in 0..shift {
        output[i] = input[half_n + i];
    }
    for i in 0..half_n {
        output[shift + i] = input[i];
    }
}

/// Shift zero-frequency component to center (f64)
pub unsafe fn fftshift_c128(input: &[Complex128], output: &mut [Complex128]) {
    let n = input.len();
    let half_n = n / 2;

    for i in 0..half_n {
        output[i] = input[half_n + i];
    }
    for i in 0..n - half_n {
        output[half_n + i] = input[i];
    }
}

/// Inverse shift (f64)
pub unsafe fn ifftshift_c128(input: &[Complex128], output: &mut [Complex128]) {
    let n = input.len();
    let half_n = (n + 1) / 2;

    let shift = n - half_n;
    for i in 0..shift {
        output[i] = input[half_n + i];
    }
    for i in 0..half_n {
        output[shift + i] = input[i];
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

    #[test]
    fn test_rfft() {
        // Real FFT of [1, 2, 3, 4]
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [Complex64::default(); 3]; // N/2 + 1

        unsafe {
            rfft_c64(&input, &mut output, 1.0);
        }

        // Expected (from numpy.fft.rfft):
        // [10+0j, -2+2j, -2+0j]
        assert!((output[0].re - 10.0).abs() < 1e-4);
        assert!(output[0].im.abs() < 1e-4);
        assert!((output[1].re - (-2.0)).abs() < 1e-4);
        assert!((output[1].im - 2.0).abs() < 1e-4);
        assert!((output[2].re - (-2.0)).abs() < 1e-4);
        assert!(output[2].im.abs() < 1e-4);
    }

    #[test]
    fn test_irfft_roundtrip() {
        let original = [1.0f32, 2.0, 3.0, 4.0];
        let mut rfft_out = [Complex64::default(); 3];
        let mut recovered = [0.0f32; 4];

        unsafe {
            rfft_c64(&original, &mut rfft_out, 1.0);
            irfft_c64(&rfft_out, &mut recovered, 0.25); // normalize by 1/N
        }

        for i in 0..4 {
            assert!(
                (recovered[i] - original[i]).abs() < 1e-4,
                "Mismatch at {}: {} vs {}",
                i,
                recovered[i],
                original[i]
            );
        }
    }

    #[test]
    fn test_fftshift() {
        let input = [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        let mut output = [Complex64::default(); 4];

        unsafe {
            fftshift_c64(&input, &mut output);
        }

        // [0, 1, 2, 3] -> [2, 3, 0, 1]
        assert!((output[0].re - 2.0).abs() < 1e-5);
        assert!((output[1].re - 3.0).abs() < 1e-5);
        assert!((output[2].re - 0.0).abs() < 1e-5);
        assert!((output[3].re - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_fftshift_ifftshift_roundtrip() {
        let original = [
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(7.0, 8.0),
        ];
        let mut shifted = [Complex64::default(); 4];
        let mut unshifted = [Complex64::default(); 4];

        unsafe {
            fftshift_c64(&original, &mut shifted);
            ifftshift_c64(&shifted, &mut unshifted);
        }

        for i in 0..4 {
            assert!((unshifted[i].re - original[i].re).abs() < 1e-5);
            assert!((unshifted[i].im - original[i].im).abs() < 1e-5);
        }
    }
}
