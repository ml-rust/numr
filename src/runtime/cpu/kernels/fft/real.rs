//! One-shot real FFT entry points (rfft/irfft) and their kernel tests.
//!
//! Each function builds a plan from [`super::real_plan`] for a single row.
//! Runtime code builds the plan once per batch, so this module is test-only.

use crate::dtype::{Complex64, Complex128};

use super::real_plan::{IrfftPlanC64, IrfftPlanC128, RfftPlanC64, RfftPlanC128};

/// Real-to-complex FFT (f32 precision)
///
/// Power-of-two N >= 2 uses the half-size packing trick. Every other size
/// falls back to a full complex Bluestein transform, keeping the first
/// N/2 + 1 bins.
///
/// # Safety
///
/// * N must be >= 1
/// * `output` must have length N/2 + 1
pub(super) unsafe fn rfft_c64(input: &[f32], output: &mut [Complex64], normalize_factor: f32) {
    RfftPlanC64::new(input.len()).execute(input, output, normalize_factor);
}

/// Complex-to-real inverse FFT (f32 precision)
///
/// The output length is authoritative: N is taken from `output`, so odd N (which
/// cannot be recovered from `input.len()`) is handled correctly.
///
/// # Safety
///
/// * `input` must have length N/2 + 1
/// * `output` must have length N >= 1
pub(super) unsafe fn irfft_c64(input: &[Complex64], output: &mut [f32], normalize_factor: f32) {
    IrfftPlanC64::new(output.len()).execute(input, output, normalize_factor);
}

/// Real-to-complex FFT (f64 precision)
///
/// # Safety
///
/// * N must be >= 1
/// * `output` must have length N/2 + 1
pub(super) unsafe fn rfft_c128(input: &[f64], output: &mut [Complex128], normalize_factor: f64) {
    RfftPlanC128::new(input.len()).execute(input, output, normalize_factor);
}

/// Complex-to-real inverse FFT (f64 precision)
///
/// # Safety
///
/// * `input` must have length N/2 + 1
/// * `output` must have length N >= 1
pub(super) unsafe fn irfft_c128(input: &[Complex128], output: &mut [f64], normalize_factor: f64) {
    IrfftPlanC128::new(output.len()).execute(input, output, normalize_factor);
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;

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
    fn test_rfft_arbitrary_size_matches_naive_dft() {
        for &n in &ARBITRARY_SIZES {
            let real_f64: Vec<f64> = deterministic_samples(n, 0xabcd_0001 ^ n as u64)
                .iter()
                .map(|c| c.re)
                .collect();
            let as_complex: Vec<Complex128> =
                real_f64.iter().map(|&x| Complex128::new(x, 0.0)).collect();
            let full = naive_dft(&as_complex, false);
            let expected = &full[..n / 2 + 1];
            let scale = signal_scale(&as_complex);

            let mut out_c128 = vec![Complex128::default(); n / 2 + 1];
            unsafe {
                rfft_c128(&real_f64, &mut out_c128, 1.0);
            }
            assert_close_c128(
                &out_c128,
                expected,
                1e-11 * scale + 1e-11,
                &format!("rfft c128 n={}", n),
            );

            let real_f32: Vec<f32> = real_f64.iter().map(|&x| x as f32).collect();
            let as_complex_f32: Vec<Complex128> = real_f32
                .iter()
                .map(|&x| Complex128::new(x as f64, 0.0))
                .collect();
            let full_f32 = naive_dft(&as_complex_f32, false);
            let expected_f32 = &full_f32[..n / 2 + 1];

            let mut out_c64 = vec![Complex64::default(); n / 2 + 1];
            unsafe {
                rfft_c64(&real_f32, &mut out_c64, 1.0);
            }
            assert_close_c64(
                &out_c64,
                expected_f32,
                1e-6 * scale + 1e-5,
                &format!("rfft c64 n={}", n),
            );
        }
    }

    #[test]
    fn test_rfft_irfft_roundtrip_400() {
        let n = 400;
        let original_f64: Vec<f64> = deterministic_samples(n, 0x1122_3344)
            .iter()
            .map(|c| c.re)
            .collect();

        let mut spectrum_c128 = vec![Complex128::default(); n / 2 + 1];
        let mut recovered_f64 = vec![0.0f64; n];
        unsafe {
            rfft_c128(&original_f64, &mut spectrum_c128, 1.0);
            irfft_c128(&spectrum_c128, &mut recovered_f64, 1.0 / n as f64);
        }
        for i in 0..n {
            assert!(
                (recovered_f64[i] - original_f64[i]).abs() < 1e-12,
                "c128 sample {}: got {}, want {}",
                i,
                recovered_f64[i],
                original_f64[i]
            );
        }

        let original_f32: Vec<f32> = original_f64.iter().map(|&x| x as f32).collect();
        let mut spectrum_c64 = vec![Complex64::default(); n / 2 + 1];
        let mut recovered_f32 = vec![0.0f32; n];
        unsafe {
            rfft_c64(&original_f32, &mut spectrum_c64, 1.0);
            irfft_c64(&spectrum_c64, &mut recovered_f32, 1.0 / n as f32);
        }
        for i in 0..n {
            assert!(
                (recovered_f32[i] - original_f32[i]).abs() < 1e-5,
                "c64 sample {}: got {}, want {}",
                i,
                recovered_f32[i],
                original_f32[i]
            );
        }
    }

    #[test]
    fn test_rfft_irfft_roundtrip_odd_sizes() {
        // Odd N cannot be inferred from the spectrum length, so the kernels take
        // N from the output slice. Cover both parities of `N/2`.
        for &n in &[3usize, 5, 7, 101] {
            let original: Vec<f64> = deterministic_samples(n, 0x9988_7766 ^ n as u64)
                .iter()
                .map(|c| c.re)
                .collect();

            let mut spectrum = vec![Complex128::default(); n / 2 + 1];
            let mut recovered = vec![0.0f64; n];
            unsafe {
                rfft_c128(&original, &mut spectrum, 1.0);
                irfft_c128(&spectrum, &mut recovered, 1.0 / n as f64);
            }

            for i in 0..n {
                assert!(
                    (recovered[i] - original[i]).abs() < 1e-12,
                    "n={} sample {}: got {}, want {}",
                    n,
                    i,
                    recovered[i],
                    original[i]
                );
            }
        }
    }

    /// Power-of-two rfft with the unpack twiddle evaluated inline per bin, as
    /// the kernel did before the plan cached it.
    fn rfft_c64_inline_reference(input: &[f32]) -> Vec<Complex64> {
        use super::super::stockham::stockham_fft_c64;
        let n = input.len();
        let half_n = n / 2;
        let packed: Vec<Complex64> = (0..half_n)
            .map(|k| Complex64::new(input[2 * k], input[2 * k + 1]))
            .collect();
        let mut z = vec![Complex64::default(); half_n];
        unsafe { stockham_fft_c64(&packed, &mut z, false, 1.0) };
        let mut out = vec![Complex64::default(); half_n + 1];
        out[0] = Complex64::new(z[0].re + z[0].im, 0.0);
        for k in 1..half_n {
            let z_k = z[k];
            let z_nk = z[half_n - k].conj();
            let x_even = (z_k + z_nk) * Complex64::new(0.5, 0.0);
            let x_odd = (z_k - z_nk) * Complex64::new(0.0, -0.5);
            let theta = -2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
            let twiddle = Complex64::new(theta.cos() as f32, theta.sin() as f32);
            out[k] = x_even + x_odd * twiddle;
        }
        out[half_n] = Complex64::new(z[0].re - z[0].im, 0.0);
        out
    }

    #[test]
    fn test_rfft_plan_bit_identical_to_inline_unpack() {
        for &n in &[2usize, 4, 16, 256, 1024] {
            let x: Vec<f32> = deterministic_samples(n, 0x4242 ^ n as u64)
                .iter()
                .map(|c| c.re as f32)
                .collect();
            let want = rfft_c64_inline_reference(&x);
            let mut got = vec![Complex64::default(); n / 2 + 1];
            unsafe { rfft_c64(&x, &mut got, 1.0) };
            for k in 0..=n / 2 {
                assert_eq!(got[k].re.to_bits(), want[k].re.to_bits(), "n={n} bin {k}");
                assert_eq!(got[k].im.to_bits(), want[k].im.to_bits(), "n={n} bin {k}");
            }
        }
    }

    /// One plan reused across rows gives every row the bits of a one-shot call.
    #[test]
    fn test_real_plans_reused_across_rows_are_bit_identical() {
        for &n in &[1usize, 2, 7, 64, 400, 512] {
            let rfft64 = RfftPlanC64::new(n);
            let rfft128 = RfftPlanC128::new(n);
            let irfft64 = IrfftPlanC64::new(n);
            let irfft128 = IrfftPlanC128::new(n);
            let bins = n / 2 + 1;
            for row in 0..3u64 {
                let wide: Vec<f64> = deterministic_samples(n, 0x5150 ^ (n as u64) ^ (row << 32))
                    .iter()
                    .map(|c| c.re)
                    .collect();
                let narrow: Vec<f32> = wide.iter().map(|&v| v as f32).collect();

                let (mut a64, mut b64) = (
                    vec![Complex64::default(); bins],
                    vec![Complex64::default(); bins],
                );
                let (mut a128, mut b128) = (
                    vec![Complex128::default(); bins],
                    vec![Complex128::default(); bins],
                );
                let (mut r64, mut s64) = (vec![0.0f32; n], vec![0.0f32; n]);
                let (mut r128, mut s128) = (vec![0.0f64; n], vec![0.0f64; n]);
                unsafe {
                    rfft64.execute(&narrow, &mut a64, 0.5);
                    rfft_c64(&narrow, &mut b64, 0.5);
                    rfft128.execute(&wide, &mut a128, 0.5);
                    rfft_c128(&wide, &mut b128, 0.5);
                    irfft64.execute(&a64, &mut r64, 0.25);
                    irfft_c64(&a64, &mut s64, 0.25);
                    irfft128.execute(&a128, &mut r128, 0.25);
                    irfft_c128(&a128, &mut s128, 0.25);
                }
                for k in 0..bins {
                    assert_eq!(
                        a64[k].re.to_bits(),
                        b64[k].re.to_bits(),
                        "rfft c64 n={n} bin {k}"
                    );
                    assert_eq!(
                        a64[k].im.to_bits(),
                        b64[k].im.to_bits(),
                        "rfft c64 n={n} bin {k}"
                    );
                    assert_eq!(
                        a128[k].re.to_bits(),
                        b128[k].re.to_bits(),
                        "rfft c128 n={n} bin {k}"
                    );
                    assert_eq!(
                        a128[k].im.to_bits(),
                        b128[k].im.to_bits(),
                        "rfft c128 n={n} bin {k}"
                    );
                }
                for i in 0..n {
                    assert_eq!(r64[i].to_bits(), s64[i].to_bits(), "irfft c64 n={n} i={i}");
                    assert_eq!(
                        r128[i].to_bits(),
                        s128[i].to_bits(),
                        "irfft c128 n={n} i={i}"
                    );
                }
            }
        }
    }
}
