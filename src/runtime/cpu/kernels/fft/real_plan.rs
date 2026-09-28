//! Real FFT plans (rfft/irfft) for one transform length.
//!
//! A plan holds every table a row needs: the half-size Stockham twiddles and
//! the unpack twiddles `W_N^(-k)` for power-of-two N, or a Bluestein plan for
//! any other N. A batch of rows of one length builds the plan once.

use crate::dtype::{Complex64, Complex128};
use std::f64::consts::PI;

use super::bluestein::BluesteinPlan;
use super::dispatch::{FftPlanC64, FftPlanC128};
use super::stockham::{stockham_fft_c64_with, stockham_fft_c128_with};
use super::twiddles::StockhamTwiddles;

/// Unpack twiddle angle for bin `k` of an N-point real transform.
#[inline(always)]
fn unpack_theta(k: usize, n: usize) -> f64 {
    -2.0 * PI * (k as f64) / (n as f64)
}

/// True when the packed half-size path applies.
#[inline(always)]
fn uses_packing(n: usize) -> bool {
    n >= 2 && n.is_power_of_two()
}

enum RfftKindC64 {
    /// `unpack[k]` holds `W_N^(-k)` for `k` in `1..N/2`; entry 0 is unused.
    Packed {
        half: StockhamTwiddles<Complex64>,
        unpack: Vec<Complex64>,
    },
    Bluestein(BluesteinPlan),
}

/// Real-to-complex FFT plan, f32 input.
///
/// For power-of-two N the N reals are packed as N/2 complex values
/// `z[k] = x[2k] + i*x[2k+1]`, transformed at half size, and unpacked into
/// N/2 + 1 bins. Every other size runs a full complex Bluestein transform and
/// keeps the first N/2 + 1 bins.
pub struct RfftPlanC64 {
    n: usize,
    kind: RfftKindC64,
}

impl RfftPlanC64 {
    /// Build the plan for N real inputs.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn new(n: usize) -> Self {
        let kind = if uses_packing(n) {
            let half_n = n / 2;
            let mut unpack = vec![Complex64::default(); half_n];
            for (k, w) in unpack.iter_mut().enumerate().skip(1) {
                let theta = unpack_theta(k, n);
                *w = Complex64::new(theta.cos() as f32, theta.sin() as f32);
            }
            RfftKindC64::Packed {
                half: StockhamTwiddles::new_c64(half_n, false),
                unpack,
            }
        } else {
            RfftKindC64::Bluestein(BluesteinPlan::new(n, false))
        };
        Self { n, kind }
    }

    /// Transform one row.
    ///
    /// # Safety
    ///
    /// * `input` must have length N and `output` length N/2 + 1
    pub unsafe fn execute(&self, input: &[f32], output: &mut [Complex64], normalize_factor: f32) {
        let n = self.n;
        debug_assert_eq!(input.len(), n);
        debug_assert_eq!(output.len(), n / 2 + 1);

        let (half, unpack) = match &self.kind {
            RfftKindC64::Packed { half, unpack } => (half, unpack),
            RfftKindC64::Bluestein(plan) => {
                plan.execute_rfft_f32(input, output, normalize_factor);
                return;
            }
        };
        let half_n = n / 2;

        // Step 1: Pack real values into complex
        let mut packed: Vec<Complex64> = Vec::with_capacity(half_n);
        for k in 0..half_n {
            packed.push(Complex64::new(input[2 * k], input[2 * k + 1]));
        }

        // Step 2: Compute half-size complex FFT (no normalization yet)
        let mut fft_result = vec![Complex64::default(); half_n];
        stockham_fft_c64_with(&packed, &mut fft_result, half, 1.0);

        // Step 3: Unpack to get full rfft output
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

            let result = x_even + x_odd * unpack[k];
            output[k] = Complex64::new(result.re * normalize_factor, result.im * normalize_factor);
        }

        // Nyquist component (k = N/2)
        output[half_n] = Complex64::new(
            (fft_result[0].re - fft_result[0].im) * normalize_factor,
            0.0,
        );
    }
}

enum RfftKindC128 {
    /// `unpack[k]` holds `W_N^(-k)` for `k` in `1..N/2`; entry 0 is unused.
    Packed {
        half: StockhamTwiddles<Complex128>,
        unpack: Vec<Complex128>,
    },
    Bluestein(BluesteinPlan),
}

/// Real-to-complex FFT plan, f64 input. Same algorithm as [`RfftPlanC64`].
pub struct RfftPlanC128 {
    n: usize,
    kind: RfftKindC128,
}

impl RfftPlanC128 {
    /// Build the plan for N real inputs.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn new(n: usize) -> Self {
        let kind = if uses_packing(n) {
            let half_n = n / 2;
            let mut unpack = vec![Complex128::default(); half_n];
            for (k, w) in unpack.iter_mut().enumerate().skip(1) {
                let theta = unpack_theta(k, n);
                *w = Complex128::new(theta.cos(), theta.sin());
            }
            RfftKindC128::Packed {
                half: StockhamTwiddles::new_c128(half_n, false),
                unpack,
            }
        } else {
            RfftKindC128::Bluestein(BluesteinPlan::new(n, false))
        };
        Self { n, kind }
    }

    /// Transform one row.
    ///
    /// # Safety
    ///
    /// * `input` must have length N and `output` length N/2 + 1
    pub unsafe fn execute(&self, input: &[f64], output: &mut [Complex128], normalize_factor: f64) {
        let n = self.n;
        debug_assert_eq!(input.len(), n);
        debug_assert_eq!(output.len(), n / 2 + 1);

        let (half, unpack) = match &self.kind {
            RfftKindC128::Packed { half, unpack } => (half, unpack),
            RfftKindC128::Bluestein(plan) => {
                plan.execute_rfft_f64(input, output, normalize_factor);
                return;
            }
        };
        let half_n = n / 2;

        let mut packed: Vec<Complex128> = Vec::with_capacity(half_n);
        for k in 0..half_n {
            packed.push(Complex128::new(input[2 * k], input[2 * k + 1]));
        }

        let mut fft_result = vec![Complex128::default(); half_n];
        stockham_fft_c128_with(&packed, &mut fft_result, half, 1.0);

        output[0] = Complex128::new(
            (fft_result[0].re + fft_result[0].im) * normalize_factor,
            0.0,
        );

        for k in 1..half_n {
            let z_k = fft_result[k];
            let z_nk = fft_result[half_n - k].conj();

            let x_even = (z_k + z_nk) * Complex128::new(0.5, 0.0);
            let x_odd = (z_k - z_nk) * Complex128::new(0.0, -0.5);

            let result = x_even + x_odd * unpack[k];
            output[k] = Complex128::new(result.re * normalize_factor, result.im * normalize_factor);
        }

        output[half_n] = Complex128::new(
            (fft_result[0].re - fft_result[0].im) * normalize_factor,
            0.0,
        );
    }
}

/// Complex-to-real inverse FFT plan, f32 output.
///
/// Extends the Hermitian-symmetric N/2 + 1 bins to the full spectrum, runs an
/// N-point inverse transform, and keeps the real parts. N is the output length,
/// so odd N is handled.
pub struct IrfftPlanC64 {
    fft: FftPlanC64,
}

impl IrfftPlanC64 {
    /// Build the plan for N real outputs.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn new(n: usize) -> Self {
        Self {
            fft: FftPlanC64::new(n, true),
        }
    }

    /// Transform one row.
    ///
    /// # Safety
    ///
    /// * `input` must have length N/2 + 1 and `output` length N
    pub unsafe fn execute(&self, input: &[Complex64], output: &mut [f32], normalize_factor: f32) {
        let n = self.fft.n();
        debug_assert_eq!(output.len(), n);
        let half_n = n / 2;
        debug_assert_eq!(input.len(), half_n + 1);

        // For even N the Nyquist bin (k == N - k) is stored once, without conjugation.
        let mut full_spectrum = vec![Complex64::default(); n];
        full_spectrum[0] = input[0];
        for k in 1..=half_n {
            full_spectrum[k] = input[k];
            if n - k != k {
                full_spectrum[n - k] = input[k].conj();
            }
        }

        let mut ifft_result = vec![Complex64::default(); n];
        self.fft
            .execute(&full_spectrum, &mut ifft_result, normalize_factor);

        for (o, c) in output.iter_mut().zip(ifft_result.iter()) {
            *o = c.re;
        }
    }
}

/// Complex-to-real inverse FFT plan, f64 output. Same algorithm as [`IrfftPlanC64`].
pub struct IrfftPlanC128 {
    fft: FftPlanC128,
}

impl IrfftPlanC128 {
    /// Build the plan for N real outputs.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub fn new(n: usize) -> Self {
        Self {
            fft: FftPlanC128::new(n, true),
        }
    }

    /// Transform one row.
    ///
    /// # Safety
    ///
    /// * `input` must have length N/2 + 1 and `output` length N
    pub unsafe fn execute(&self, input: &[Complex128], output: &mut [f64], normalize_factor: f64) {
        let n = self.fft.n();
        debug_assert_eq!(output.len(), n);
        let half_n = n / 2;
        debug_assert_eq!(input.len(), half_n + 1);

        let mut full_spectrum = vec![Complex128::default(); n];
        full_spectrum[0] = input[0];
        for k in 1..=half_n {
            full_spectrum[k] = input[k];
            if n - k != k {
                full_spectrum[n - k] = input[k].conj();
            }
        }

        let mut ifft_result = vec![Complex128::default(); n];
        self.fft
            .execute(&full_spectrum, &mut ifft_result, normalize_factor);

        for (o, c) in output.iter_mut().zip(ifft_result.iter()) {
            *o = c.re;
        }
    }
}
