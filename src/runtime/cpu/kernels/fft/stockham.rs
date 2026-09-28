//! Stockham autosort radix-2 FFT kernels (power-of-two sizes)

use crate::dtype::{Complex64, Complex128};

use super::twiddles::StockhamTwiddles;

// ============================================================================
// Complex64 (f32) FFT Kernels
// ============================================================================

/// Stockham FFT for Complex64 data
///
/// Builds the twiddle table for this call. A caller that transforms many rows
/// of the same length builds one [`StockhamTwiddles`] and calls
/// [`stockham_fft_c64_with`] per row instead.
///
/// # Safety
///
/// * `input` and `output` must be valid slices of length N
/// * N must be a power of 2
#[cfg(test)]
pub(super) unsafe fn stockham_fft_c64(
    input: &[Complex64],
    output: &mut [Complex64],
    inverse: bool,
    normalize_factor: f32,
) {
    let twiddles = StockhamTwiddles::new_c64(input.len(), inverse);
    stockham_fft_c64_with(input, output, &twiddles, normalize_factor);
}

/// Stockham FFT for Complex64 data with a prebuilt twiddle table.
///
/// The direction is the one `twiddles` was built for (its sign is in the table).
///
/// # Safety
///
/// * `input` and `output` must be valid slices of length N
/// * N must be a power of 2 and equal `twiddles.n()`
pub(super) unsafe fn stockham_fft_c64_with(
    input: &[Complex64],
    output: &mut [Complex64],
    twiddles: &StockhamTwiddles<Complex64>,
    normalize_factor: f32,
) {
    let n = input.len();
    debug_assert!(n > 0 && (n & (n - 1)) == 0, "N must be power of 2");
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(twiddles.n(), n);

    if n == 1 {
        output[0] = Complex64::new(
            input[0].re * normalize_factor,
            input[0].im * normalize_factor,
        );
        return;
    }

    let log_n = n.trailing_zeros() as usize;

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
        // Twiddle factor: W_m^b = exp(sign * 2πi * b / m), indexed by b
        let stage_twiddles = twiddles.stage(half_m);

        // Process all butterflies in this stage
        for g in 0..groups {
            for (b, &twiddle) in stage_twiddles.iter().enumerate() {
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

// ============================================================================
// Complex128 (f64) FFT Kernels
// ============================================================================

/// Stockham FFT for Complex128 data
///
/// Builds the twiddle table for this call. See [`stockham_fft_c128_with`] to
/// reuse one table across rows.
///
/// # Safety
///
/// * `input` and `output` must be valid slices of length N
/// * N must be a power of 2
#[cfg(test)]
pub(super) unsafe fn stockham_fft_c128(
    input: &[Complex128],
    output: &mut [Complex128],
    inverse: bool,
    normalize_factor: f64,
) {
    let twiddles = StockhamTwiddles::new_c128(input.len(), inverse);
    stockham_fft_c128_with(input, output, &twiddles, normalize_factor);
}

/// Stockham FFT for Complex128 data with a prebuilt twiddle table.
///
/// # Safety
///
/// * `input` and `output` must be valid slices of length N
/// * N must be a power of 2 and equal `twiddles.n()`
pub(super) unsafe fn stockham_fft_c128_with(
    input: &[Complex128],
    output: &mut [Complex128],
    twiddles: &StockhamTwiddles<Complex128>,
    normalize_factor: f64,
) {
    let n = input.len();
    debug_assert!(n > 0 && (n & (n - 1)) == 0, "N must be power of 2");
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(twiddles.n(), n);

    if n == 1 {
        output[0] = Complex128::new(
            input[0].re * normalize_factor,
            input[0].im * normalize_factor,
        );
        return;
    }

    let log_n = n.trailing_zeros() as usize;

    // Double buffering
    let mut buf_a: Vec<Complex128> = input.to_vec();
    let mut buf_b: Vec<Complex128> = vec![Complex128::default(); n];

    let mut src = &mut buf_a;
    let mut dst = &mut buf_b;

    for stage in 0..log_n {
        let m = 1 << (stage + 1);
        let half_m = 1 << stage;
        let groups = n / m;
        let stage_twiddles = twiddles.stage(half_m);

        for g in 0..groups {
            for (b, &twiddle) in stage_twiddles.iter().enumerate() {
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

    /// Butterfly loop that evaluates each twiddle inline, as the kernel did
    /// before the per-stage table. Generic over the complex type through `tw`.
    fn inline_twiddle_reference<C>(input: &[C], inverse: bool, tw: impl Fn(f64) -> C) -> Vec<C>
    where
        C: Copy
            + Default
            + std::ops::Add<Output = C>
            + std::ops::Sub<Output = C>
            + std::ops::Mul<Output = C>,
    {
        let n = input.len();
        let sign = if inverse { 1.0f64 } else { -1.0f64 };
        let mut src = input.to_vec();
        let mut dst = vec![C::default(); n];
        for stage in 0..n.trailing_zeros() as usize {
            let m: usize = 1 << (stage + 1);
            let half_m: usize = 1 << stage;
            for g in 0..n / m {
                for b in 0..half_m {
                    let theta = sign * 2.0 * std::f64::consts::PI * (b as f64) / (m as f64);
                    let twiddle = tw(theta);
                    let even = src[g * half_m + b];
                    let odd = src[n / 2 + g * half_m + b] * twiddle;
                    dst[g * m + b] = even + odd;
                    dst[g * m + b + half_m] = even - odd;
                }
            }
            std::mem::swap(&mut src, &mut dst);
        }
        src
    }

    #[test]
    fn test_twiddle_table_is_bit_identical_to_inline_twiddles() {
        for &n in &[2usize, 4, 8, 64, 256, 1024] {
            let wide: Vec<Complex128> = (0..n)
                .map(|i| {
                    let x = i as f64;
                    Complex128::new((x * 0.37).sin() + 0.1 * x, (x * 1.13).cos() - 0.05 * x)
                })
                .collect();
            let narrow: Vec<Complex64> = wide
                .iter()
                .map(|c| Complex64::new(c.re as f32, c.im as f32))
                .collect();
            for &inverse in &[false, true] {
                let want64 = inline_twiddle_reference(&narrow, inverse, |t| {
                    Complex64::new(t.cos() as f32, t.sin() as f32)
                });
                let mut got64 = vec![Complex64::default(); n];
                unsafe { stockham_fft_c64(&narrow, &mut got64, inverse, 1.0) };
                for i in 0..n {
                    assert_eq!(
                        got64[i].re.to_bits(),
                        want64[i].re.to_bits(),
                        "c64 n={n} i={i}"
                    );
                    assert_eq!(
                        got64[i].im.to_bits(),
                        want64[i].im.to_bits(),
                        "c64 n={n} i={i}"
                    );
                }

                let want128 =
                    inline_twiddle_reference(&wide, inverse, |t| Complex128::new(t.cos(), t.sin()));
                let mut got128 = vec![Complex128::default(); n];
                unsafe { stockham_fft_c128(&wide, &mut got128, inverse, 1.0) };
                for i in 0..n {
                    assert_eq!(
                        got128[i].re.to_bits(),
                        want128[i].re.to_bits(),
                        "c128 n={n} i={i}"
                    );
                    assert_eq!(
                        got128[i].im.to_bits(),
                        want128[i].im.to_bits(),
                        "c128 n={n} i={i}"
                    );
                }
            }
        }
    }
}
