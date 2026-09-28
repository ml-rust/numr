//! Size-dispatching FFT plans and entry points.
//!
//! Power-of-two sizes take the Stockham path unchanged; every other size uses
//! Bluestein's algorithm built on that same Stockham kernel. A plan holds the
//! per-size tables, so a batch of rows of one length builds them once.

use super::bluestein::BluesteinPlan;
use super::stockham::{stockham_fft_c64_with, stockham_fft_c128_with};
use super::twiddles::StockhamTwiddles;
use crate::dtype::{Complex64, Complex128};

enum PlanKindC64 {
    Stockham(StockhamTwiddles<Complex64>),
    Bluestein(BluesteinPlan),
}

/// Complex64 FFT plan for one size `N >= 1` and one direction.
pub(super) struct FftPlanC64 {
    n: usize,
    kind: PlanKindC64,
}

impl FftPlanC64 {
    /// Build the tables for an N-point transform.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub(super) fn new(n: usize, inverse: bool) -> Self {
        let kind = if n.is_power_of_two() {
            PlanKindC64::Stockham(StockhamTwiddles::new_c64(n, inverse))
        } else {
            PlanKindC64::Bluestein(BluesteinPlan::new(n, inverse))
        };
        Self { n, kind }
    }

    /// Transform length.
    pub(super) fn n(&self) -> usize {
        self.n
    }

    /// Transform one row.
    ///
    /// # Safety
    ///
    /// * `input` and `output` must both have length `self.n()`
    pub(super) unsafe fn execute(&self, input: &[Complex64], output: &mut [Complex64], nf: f32) {
        debug_assert_eq!(input.len(), self.n);
        debug_assert_eq!(output.len(), self.n);
        match &self.kind {
            PlanKindC64::Stockham(twiddles) => stockham_fft_c64_with(input, output, twiddles, nf),
            PlanKindC64::Bluestein(plan) => plan.execute_c64(input, output, nf),
        }
    }
}

enum PlanKindC128 {
    Stockham(StockhamTwiddles<Complex128>),
    Bluestein(BluesteinPlan),
}

/// Complex128 FFT plan for one size `N >= 1` and one direction.
pub(super) struct FftPlanC128 {
    n: usize,
    kind: PlanKindC128,
}

impl FftPlanC128 {
    /// Build the tables for an N-point transform.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`.
    pub(super) fn new(n: usize, inverse: bool) -> Self {
        let kind = if n.is_power_of_two() {
            PlanKindC128::Stockham(StockhamTwiddles::new_c128(n, inverse))
        } else {
            PlanKindC128::Bluestein(BluesteinPlan::new(n, inverse))
        };
        Self { n, kind }
    }

    /// Transform length.
    pub(super) fn n(&self) -> usize {
        self.n
    }

    /// Transform one row.
    ///
    /// # Safety
    ///
    /// * `input` and `output` must both have length `self.n()`
    pub(super) unsafe fn execute(&self, input: &[Complex128], output: &mut [Complex128], nf: f64) {
        debug_assert_eq!(input.len(), self.n);
        debug_assert_eq!(output.len(), self.n);
        match &self.kind {
            PlanKindC128::Stockham(twiddles) => stockham_fft_c128_with(input, output, twiddles, nf),
            PlanKindC128::Bluestein(plan) => plan.execute_c128(input, output, nf),
        }
    }
}

/// Complex64 FFT for any size `N >= 1`. Builds a one-shot [`FftPlanC64`].
///
/// # Safety
///
/// * `input` and `output` must be valid slices of the same length `N >= 1`
#[cfg(test)]
pub(super) unsafe fn fft_c64(
    input: &[Complex64],
    output: &mut [Complex64],
    inverse: bool,
    normalize_factor: f32,
) {
    FftPlanC64::new(input.len(), inverse).execute(input, output, normalize_factor);
}

/// Complex128 FFT for any size `N >= 1`. Builds a one-shot [`FftPlanC128`].
///
/// # Safety
///
/// * `input` and `output` must be valid slices of the same length `N >= 1`
#[cfg(test)]
pub(super) unsafe fn fft_c128(
    input: &[Complex128],
    output: &mut [Complex128],
    inverse: bool,
    normalize_factor: f64,
) {
    FftPlanC128::new(input.len(), inverse).execute(input, output, normalize_factor);
}

#[cfg(test)]
mod tests {
    use super::super::stockham::{stockham_fft_c64, stockham_fft_c128};
    use super::super::test_support::*;
    use super::*;

    #[test]
    fn test_bluestein_forward_matches_naive_dft_c128() {
        for &n in &ARBITRARY_SIZES {
            let input = deterministic_samples(n, 0x5eed_1234 ^ n as u64);
            let expected = naive_dft(&input, false);
            let tol = 1e-11 * signal_scale(&input) + 1e-11;

            let mut got = vec![Complex128::default(); n];
            unsafe {
                fft_c128(&input, &mut got, false, 1.0);
            }

            assert_close_c128(&got, &expected, tol, &format!("forward c128 n={}", n));
        }
    }

    #[test]
    fn test_bluestein_inverse_matches_naive_dft_c128() {
        for &n in &ARBITRARY_SIZES {
            let input = deterministic_samples(n, 0xbeef_0000 ^ n as u64);
            let expected = naive_dft(&input, true);
            let tol = 1e-11 * signal_scale(&input) + 1e-11;

            // Unnormalized inverse, so it matches the reference sum directly.
            let mut got = vec![Complex128::default(); n];
            unsafe {
                fft_c128(&input, &mut got, true, 1.0);
            }

            assert_close_c128(&got, &expected, tol, &format!("inverse c128 n={}", n));
        }
    }

    #[test]
    fn test_bluestein_forward_matches_naive_dft_c64() {
        for &n in &ARBITRARY_SIZES {
            let wide = deterministic_samples(n, 0x1357_9bdf ^ n as u64);
            // Narrow to f32 first, then use the exact same samples for the reference
            // so the comparison isolates the transform, not the input rounding.
            let narrow: Vec<Complex64> = wide
                .iter()
                .map(|c| Complex64::new(c.re as f32, c.im as f32))
                .collect();
            let reference_input: Vec<Complex128> = narrow
                .iter()
                .map(|c| Complex128::new(c.re as f64, c.im as f64))
                .collect();
            let expected = naive_dft(&reference_input, false);
            let tol = 1e-6 * signal_scale(&reference_input) + 1e-5;

            let mut got = vec![Complex64::default(); n];
            unsafe {
                fft_c64(&narrow, &mut got, false, 1.0);
            }

            assert_close_c64(&got, &expected, tol, &format!("forward c64 n={}", n));
        }
    }

    #[test]
    fn test_bluestein_inverse_matches_naive_dft_c64() {
        for &n in &ARBITRARY_SIZES {
            let wide = deterministic_samples(n, 0x2468_ace0 ^ n as u64);
            let narrow: Vec<Complex64> = wide
                .iter()
                .map(|c| Complex64::new(c.re as f32, c.im as f32))
                .collect();
            let reference_input: Vec<Complex128> = narrow
                .iter()
                .map(|c| Complex128::new(c.re as f64, c.im as f64))
                .collect();
            let expected = naive_dft(&reference_input, true);
            let tol = 1e-6 * signal_scale(&reference_input) + 1e-5;

            let mut got = vec![Complex64::default(); n];
            unsafe {
                fft_c64(&narrow, &mut got, true, 1.0);
            }

            assert_close_c64(&got, &expected, tol, &format!("inverse c64 n={}", n));
        }
    }

    #[test]
    fn test_power_of_two_dispatch_is_bit_identical_to_stockham() {
        // The size dispatcher must not perturb the existing power-of-two path.
        for &n in &[1usize, 2, 4, 8, 16, 64, 512] {
            let wide = deterministic_samples(n, 0x0f0f_0f0f ^ n as u64);
            let narrow: Vec<Complex64> = wide
                .iter()
                .map(|c| Complex64::new(c.re as f32, c.im as f32))
                .collect();

            for &inverse in &[false, true] {
                let mut via_dispatch = vec![Complex64::default(); n];
                let mut via_stockham = vec![Complex64::default(); n];
                unsafe {
                    fft_c64(&narrow, &mut via_dispatch, inverse, 0.5);
                    stockham_fft_c64(&narrow, &mut via_stockham, inverse, 0.5);
                }
                for i in 0..n {
                    assert_eq!(
                        via_dispatch[i].re.to_bits(),
                        via_stockham[i].re.to_bits(),
                        "c64 re bits differ at n={} inverse={} bin {}",
                        n,
                        inverse,
                        i
                    );
                    assert_eq!(
                        via_dispatch[i].im.to_bits(),
                        via_stockham[i].im.to_bits(),
                        "c64 im bits differ at n={} inverse={} bin {}",
                        n,
                        inverse,
                        i
                    );
                }

                let mut wide_dispatch = vec![Complex128::default(); n];
                let mut wide_stockham = vec![Complex128::default(); n];
                unsafe {
                    fft_c128(&wide, &mut wide_dispatch, inverse, 0.5);
                    stockham_fft_c128(&wide, &mut wide_stockham, inverse, 0.5);
                }
                for i in 0..n {
                    assert_eq!(
                        wide_dispatch[i].re.to_bits(),
                        wide_stockham[i].re.to_bits(),
                        "c128 re bits differ at n={} inverse={} bin {}",
                        n,
                        inverse,
                        i
                    );
                    assert_eq!(
                        wide_dispatch[i].im.to_bits(),
                        wide_stockham[i].im.to_bits(),
                        "c128 im bits differ at n={} inverse={} bin {}",
                        n,
                        inverse,
                        i
                    );
                }
            }
        }
    }

    #[test]
    fn test_power_of_two_pinned_values() {
        // FFT of the ramp x[j] = j + 1 at N = 8.
        // Closed form: X[0] = 36, X[k] = -4 + 4i*cot(pi*k/8).
        let input: Vec<Complex64> = (0..8)
            .map(|j| Complex64::new(j as f32 + 1.0, 0.0))
            .collect();
        let mut output = vec![Complex64::default(); 8];
        unsafe {
            fft_c64(&input, &mut output, false, 1.0);
        }

        let expected: [(f32, f32); 8] = [
            (36.0, 0.0),
            (-4.0, 9.656_854),
            (-4.0, 4.0),
            (-4.0, 1.656_854_2),
            (-4.0, 0.0),
            (-4.0, -1.656_854_2),
            (-4.0, -4.0),
            (-4.0, -9.656_854),
        ];

        for (i, &(re, im)) in expected.iter().enumerate() {
            assert!(
                (output[i].re - re).abs() < 1e-4,
                "bin {} re: got {}, want {}",
                i,
                output[i].re,
                re
            );
            assert!(
                (output[i].im - im).abs() < 1e-4,
                "bin {} im: got {}, want {}",
                i,
                output[i].im,
                im
            );
        }
    }

    #[test]
    fn test_fft_ifft_roundtrip_400_c128() {
        let n = 400;
        let input = deterministic_samples(n, 0xdead_beef);
        let mut spectrum = vec![Complex128::default(); n];
        let mut recovered = vec![Complex128::default(); n];

        unsafe {
            fft_c128(&input, &mut spectrum, false, 1.0);
            fft_c128(&spectrum, &mut recovered, true, 1.0 / n as f64);
        }

        assert_close_c128(&recovered, &input, 1e-12, "fft/ifft roundtrip n=400 c128");
    }

    #[test]
    fn test_fft_ifft_roundtrip_400_c64() {
        let n = 400;
        let wide = deterministic_samples(n, 0xfeed_face);
        let input: Vec<Complex64> = wide
            .iter()
            .map(|c| Complex64::new(c.re as f32, c.im as f32))
            .collect();
        let expected: Vec<Complex128> = input
            .iter()
            .map(|c| Complex128::new(c.re as f64, c.im as f64))
            .collect();

        let mut spectrum = vec![Complex64::default(); n];
        let mut recovered = vec![Complex64::default(); n];

        unsafe {
            fft_c64(&input, &mut spectrum, false, 1.0);
            fft_c64(&spectrum, &mut recovered, true, 1.0 / n as f32);
        }

        assert_close_c64(&recovered, &expected, 1e-4, "fft/ifft roundtrip n=400 c64");
    }

    #[test]
    fn test_bluestein_size_one_via_dispatch() {
        let input = [Complex128::new(2.5, -1.25)];
        let mut output = [Complex128::default(); 1];
        unsafe {
            fft_c128(&input, &mut output, false, 1.0);
        }
        assert!((output[0].re - 2.5).abs() < 1e-15);
        assert!((output[0].im + 1.25).abs() < 1e-15);
    }
}
