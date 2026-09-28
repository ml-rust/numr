//! Batched complex FFT over contiguous rows of one length.
//!
//! One [`FftPlanC64`] / [`FftPlanC128`] is built per call and shared by every
//! row, so the twiddle and Bluestein tables are computed once per batch.

use crate::dtype::{Complex64, Complex128};

use super::dispatch::{FftPlanC64, FftPlanC128};

/// Batched FFT for Complex64 data
///
/// Power-of-two sizes use the Stockham kernel; any other size uses Bluestein's
/// algorithm. Rows run in parallel when the `rayon` feature is on and the
/// caller passes `parallel`. Every row computes the same bits either way.
///
/// # Safety
///
/// * `input` and `output` must have length `batch_size * n`
/// * n must be >= 1
#[allow(clippy::too_many_arguments)]
pub unsafe fn stockham_fft_batched_c64(
    input: &[Complex64],
    output: &mut [Complex64],
    n: usize,
    batch_size: usize,
    inverse: bool,
    normalize_factor: f32,
    min_batch_len: usize,
    parallel: bool,
) {
    debug_assert_eq!(input.len(), batch_size * n);
    debug_assert_eq!(output.len(), batch_size * n);
    if batch_size == 0 {
        return;
    }

    let plan = FftPlanC64::new(n, inverse);

    #[cfg(feature = "rayon")]
    if batch_size > 1 && parallel {
        use rayon::prelude::*;
        output
            .par_chunks_mut(n)
            .zip(input.par_chunks(n))
            .with_min_len(min_batch_len.max(1))
            .for_each(|(out_chunk, in_chunk)| {
                plan.execute(in_chunk, out_chunk, normalize_factor);
            });
        return;
    }
    #[cfg(not(feature = "rayon"))]
    let _ = (min_batch_len, parallel);

    for (out_chunk, in_chunk) in output.chunks_mut(n).zip(input.chunks(n)) {
        plan.execute(in_chunk, out_chunk, normalize_factor);
    }
}

/// Batched FFT for Complex128 data
///
/// # Safety
///
/// * `input` and `output` must have length `batch_size * n`
/// * n must be >= 1
#[allow(clippy::too_many_arguments)]
pub unsafe fn stockham_fft_batched_c128(
    input: &[Complex128],
    output: &mut [Complex128],
    n: usize,
    batch_size: usize,
    inverse: bool,
    normalize_factor: f64,
    min_batch_len: usize,
    parallel: bool,
) {
    debug_assert_eq!(input.len(), batch_size * n);
    debug_assert_eq!(output.len(), batch_size * n);
    if batch_size == 0 {
        return;
    }

    let plan = FftPlanC128::new(n, inverse);

    #[cfg(feature = "rayon")]
    if batch_size > 1 && parallel {
        use rayon::prelude::*;
        output
            .par_chunks_mut(n)
            .zip(input.par_chunks(n))
            .with_min_len(min_batch_len.max(1))
            .for_each(|(out_chunk, in_chunk)| {
                plan.execute(in_chunk, out_chunk, normalize_factor);
            });
        return;
    }
    #[cfg(not(feature = "rayon"))]
    let _ = (min_batch_len, parallel);

    for (out_chunk, in_chunk) in output.chunks_mut(n).zip(input.chunks(n)) {
        plan.execute(in_chunk, out_chunk, normalize_factor);
    }
}

#[cfg(test)]
mod tests {
    use super::super::dispatch::{fft_c64, fft_c128};
    use super::super::test_support::*;
    use super::*;

    /// A shared plan must give each row the bits a one-shot transform gives it.
    #[test]
    fn test_batched_rows_bit_identical_to_one_shot() {
        for &n in &[1usize, 8, 64, 12, 400] {
            let rows = 5;
            let wide = deterministic_samples(n * rows, 0x7777_0000 ^ n as u64);
            let narrow: Vec<Complex64> = wide
                .iter()
                .map(|c| Complex64::new(c.re as f32, c.im as f32))
                .collect();
            let schedules = [(false, false), (false, true), (true, false), (true, true)];
            for (inverse, parallel) in schedules {
                let mut got64 = vec![Complex64::default(); n * rows];
                let mut got128 = vec![Complex128::default(); n * rows];
                unsafe {
                    stockham_fft_batched_c64(
                        &narrow, &mut got64, n, rows, inverse, 0.5, 1, parallel,
                    );
                    stockham_fft_batched_c128(
                        &wide,
                        &mut got128,
                        n,
                        rows,
                        inverse,
                        0.5,
                        1,
                        parallel,
                    );
                }
                for r in 0..rows {
                    let span = r * n..(r + 1) * n;
                    let mut want64 = vec![Complex64::default(); n];
                    let mut want128 = vec![Complex128::default(); n];
                    unsafe {
                        fft_c64(&narrow[span.clone()], &mut want64, inverse, 0.5);
                        fft_c128(&wide[span.clone()], &mut want128, inverse, 0.5);
                    }
                    for (i, (g, w)) in got64[span.clone()].iter().zip(&want64).enumerate() {
                        assert_eq!(g.re.to_bits(), w.re.to_bits(), "c64 n={n} row={r} i={i}");
                        assert_eq!(g.im.to_bits(), w.im.to_bits(), "c64 n={n} row={r} i={i}");
                    }
                    for (i, (g, w)) in got128[span].iter().zip(&want128).enumerate() {
                        assert_eq!(g.re.to_bits(), w.re.to_bits(), "c128 n={n} row={r} i={i}");
                        assert_eq!(g.im.to_bits(), w.im.to_bits(), "c128 n={n} row={r} i={i}");
                    }
                }
            }
        }
    }
}
