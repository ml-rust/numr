//! Shared test helpers for the FFT kernel tests (dispatch, rfft/irfft).

#![cfg(test)]

use crate::dtype::{Complex64, Complex128};
use std::f64::consts::PI;

/// Sizes exercised by every arbitrary-size test. 400 is Whisper's `n_fft`.
pub(super) const ARBITRARY_SIZES: [usize; 7] = [3, 5, 7, 12, 100, 400, 1000];

/// Deterministic xorshift64 sample generator in [-1, 1)
pub(super) fn deterministic_samples(n: usize, seed: u64) -> Vec<Complex128> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    (0..n).map(|_| Complex128::new(next(), next())).collect()
}

/// Naive O(n^2) DFT reference, evaluated in f64.
///
/// The angle uses `(j * k) % n` so the phase stays exact for large indices.
pub(super) fn naive_dft(input: &[Complex128], inverse: bool) -> Vec<Complex128> {
    let n = input.len();
    let sign = if inverse { 1.0f64 } else { -1.0f64 };
    (0..n)
        .map(|k| {
            let mut acc = Complex128::new(0.0, 0.0);
            for (j, x) in input.iter().enumerate() {
                let theta = sign * 2.0 * PI * ((j * k) % n) as f64 / (n as f64);
                acc += *x * Complex128::new(theta.cos(), theta.sin());
            }
            acc
        })
        .collect()
}

/// Magnitude scale of a signal, used to size tolerances.
pub(super) fn signal_scale(input: &[Complex128]) -> f64 {
    input.iter().map(|c| c.re.abs() + c.im.abs()).sum()
}

pub(super) fn assert_close_c128(got: &[Complex128], want: &[Complex128], tol: f64, label: &str) {
    assert_eq!(got.len(), want.len(), "{}: length mismatch", label);
    for i in 0..got.len() {
        let dr = (got[i].re - want[i].re).abs();
        let di = (got[i].im - want[i].im).abs();
        assert!(
            dr <= tol && di <= tol,
            "{}: bin {} got ({}, {}), want ({}, {}), tol {}",
            label,
            i,
            got[i].re,
            got[i].im,
            want[i].re,
            want[i].im,
            tol
        );
    }
}

pub(super) fn assert_close_c64(got: &[Complex64], want: &[Complex128], tol: f64, label: &str) {
    assert_eq!(got.len(), want.len(), "{}: length mismatch", label);
    for i in 0..got.len() {
        let dr = (got[i].re as f64 - want[i].re).abs();
        let di = (got[i].im as f64 - want[i].im).abs();
        assert!(
            dr <= tol && di <= tol,
            "{}: bin {} got ({}, {}), want ({}, {}), tol {}",
            label,
            i,
            got[i].re,
            got[i].im,
            want[i].re,
            want[i].im,
            tol
        );
    }
}
