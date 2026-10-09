//! Pairwise distance kernels (cdist, pdist).
//!
//! The unit of work is one output row segment: a block of columns of one cdist
//! row, or one pdist row of the condensed vector. The ops layer drives
//! [`cdist_block_kernel`] and [`pdist_row_kernel`] in parallel. Each output
//! element comes from one [`distance`] call on fixed inputs. The result does
//! not depend on which thread runs a unit or on the thread count.
//!
//! The kernels accept any float element. The CPU ops layer converts F16, BF16
//! and FP8 inputs once to F32 and runs these kernels on F32. A direct caller
//! with a narrow element gets the f32 accumulator from [`DistAcc`] and the
//! sequential loop.

use super::acc::DistAcc;
use super::metrics;
use crate::dtype::{DType, Element};
use crate::ops::DistanceMetric;
use num_traits::{Float, FromPrimitive};

/// Compute `out[row * m + j]` for `j` in `col_start..col_end` (one cdist unit).
///
/// # Safety
///
/// - `x` must point to valid data of length `(row + 1) * d` or more
/// - `y` must point to valid data of length `m * d`
/// - `out` must point to valid memory of length `n * m`, with `row < n`
/// - `col_start <= col_end <= m`
/// - No other thread can write `out[row * m + col_start..row * m + col_end]`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn cdist_block_kernel<T: Element + Float + FromPrimitive>(
    x: *const T,
    y: *const T,
    out: *mut T,
    row: usize,
    col_start: usize,
    col_end: usize,
    m: usize,
    d: usize,
    metric: DistanceMetric,
) {
    if T::DTYPE == DType::F64 {
        cdist_block_acc::<T, f64>(x, y, out, row, col_start, col_end, m, d, metric);
    } else {
        cdist_block_acc::<T, f32>(x, y, out, row, col_start, col_end, m, d, metric);
    }
}

/// Compute the pairs `(row, j)` for `j` in `row + 1..n` (one pdist unit).
///
/// The pairs land at condensed offset `row * (2n - row - 1) / 2 + (j - row - 1)`.
///
/// # Safety
///
/// - `x` must point to valid data of length `n * d`
/// - `out` must point to valid memory of length `n * (n - 1) / 2`
/// - `row < n`
/// - No other thread can write the pairs of `row`
#[inline]
pub unsafe fn pdist_row_kernel<T: Element + Float + FromPrimitive>(
    x: *const T,
    out: *mut T,
    n: usize,
    d: usize,
    row: usize,
    metric: DistanceMetric,
) {
    if T::DTYPE == DType::F64 {
        pdist_row_acc::<T, f64>(x, out, n, d, row, metric);
    } else {
        pdist_row_acc::<T, f32>(x, out, n, d, row, metric);
    }
}

/// One pair of rows, reduced by `metric` in accumulator `A`.
///
/// # Safety
/// `a` and `b` must each point to `d` valid elements.
#[inline]
unsafe fn distance<T: Element, A: DistAcc<T>>(
    a: *const T,
    b: *const T,
    d: usize,
    metric: DistanceMetric,
) -> A {
    match metric {
        DistanceMetric::Euclidean => metrics::euclidean::<T, A>(a, b, d),
        DistanceMetric::SquaredEuclidean => metrics::sqeuclidean::<T, A>(a, b, d),
        DistanceMetric::Manhattan => metrics::manhattan::<T, A>(a, b, d),
        DistanceMetric::Chebyshev => metrics::chebyshev::<T, A>(a, b, d),
        DistanceMetric::Minkowski(p) => metrics::minkowski::<T, A>(a, b, d, A::exponent(p)),
        DistanceMetric::Cosine => metrics::cosine::<T, A>(a, b, d),
        DistanceMetric::Correlation => metrics::correlation::<T, A>(a, b, d),
        DistanceMetric::Hamming => metrics::hamming::<T, A>(a, b, d),
        DistanceMetric::Jaccard => metrics::jaccard::<T, A>(a, b, d),
    }
}

/// One cdist unit with an explicit accumulator.
///
/// # Safety
/// Same as [`cdist_block_kernel`].
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn cdist_block_acc<T: Element, A: DistAcc<T>>(
    x: *const T,
    y: *const T,
    out: *mut T,
    row: usize,
    col_start: usize,
    col_end: usize,
    m: usize,
    d: usize,
    metric: DistanceMetric,
) {
    let a = x.add(row * d);
    let out_row = out.add(row * m);
    for j in col_start..col_end {
        let dist = distance::<T, A>(a, y.add(j * d), d, metric);
        *out_row.add(j) = dist.narrow();
    }
}

/// One pdist row with an explicit accumulator.
///
/// # Safety
/// Same as [`pdist_row_kernel`].
#[inline]
unsafe fn pdist_row_acc<T: Element, A: DistAcc<T>>(
    x: *const T,
    out: *mut T,
    n: usize,
    d: usize,
    row: usize,
    metric: DistanceMetric,
) {
    // `row * (2n - row - 1)` is even: one of the two factors always is.
    let base = row * (2 * n - row - 1) / 2;
    let a = x.add(row * d);
    for j in (row + 1)..n {
        let dist = distance::<T, A>(a, x.add(j * d), d, metric);
        *out.add(base + (j - row - 1)) = dist.narrow();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const METRICS: [DistanceMetric; 6] = [
        DistanceMetric::SquaredEuclidean,
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::Cosine,
        DistanceMetric::Chebyshev,
        DistanceMetric::Minkowski(3.0),
    ];

    fn values<T: FromPrimitive>(len: usize, seed: usize) -> Vec<T> {
        (0..len)
            .map(|i| {
                let v = (((i * 37 + seed * 11) % 251) as f64) * 0.004 - 0.5;
                T::from_f64(v).unwrap_or_else(|| panic!("value {v} out of range"))
            })
            .collect()
    }

    /// True when both slices hold the same bits. NaN compares equal to itself.
    fn same_bits<T: Element>(a: &[T], b: &[T]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| Element::to_f64(*x).to_bits() == Element::to_f64(*y).to_bits())
    }

    /// Runs cdist through `cdist_block_kernel`, one full-width block per row.
    fn serial_cdist<T: Element + Float + FromPrimitive>(
        x: &[T],
        y: &[T],
        out: &mut [T],
        n: usize,
        m: usize,
        d: usize,
        metric: DistanceMetric,
    ) {
        assert!(x.len() >= n * d && y.len() >= m * d && out.len() >= n * m);
        for row in 0..n {
            unsafe {
                cdist_block_kernel(
                    x.as_ptr(),
                    y.as_ptr(),
                    out.as_mut_ptr(),
                    row,
                    0,
                    m,
                    m,
                    d,
                    metric,
                );
            }
        }
    }

    /// Runs pdist through `pdist_row_kernel`, rows in order.
    fn serial_pdist<T: Element + Float + FromPrimitive>(
        x: &[T],
        out: &mut [T],
        n: usize,
        d: usize,
        metric: DistanceMetric,
    ) {
        assert!(x.len() >= n * d && out.len() >= n * (n - 1) / 2);
        for row in 0..n {
            unsafe {
                pdist_row_kernel(x.as_ptr(), out.as_mut_ptr(), n, d, row, metric);
            }
        }
    }

    /// One pair through `distance`, with the accumulator the kernels pick.
    fn pair_distance<T: Element + Float + FromPrimitive>(
        a: &[T],
        b: &[T],
        metric: DistanceMetric,
    ) -> T {
        assert_eq!(a.len(), b.len());
        let d = a.len();
        unsafe {
            if T::DTYPE == DType::F64 {
                distance::<T, f64>(a.as_ptr(), b.as_ptr(), d, metric).narrow()
            } else {
                distance::<T, f32>(a.as_ptr(), b.as_ptr(), d, metric).narrow()
            }
        }
    }

    /// Reference cdist: a plain double loop over pairs. Uses no block kernel.
    fn naive_cdist<T: Element + Float + FromPrimitive>(
        x: &[T],
        y: &[T],
        (n, m, d): (usize, usize, usize),
        metric: DistanceMetric,
    ) -> Vec<T> {
        let mut out = vec![<T as Element>::zero(); n * m];
        for i in 0..n {
            for j in 0..m {
                out[i * m + j] =
                    pair_distance(&x[i * d..(i + 1) * d], &y[j * d..(j + 1) * d], metric);
            }
        }
        out
    }

    /// Reference pdist: pairs `i < j` in row-major order, one running index.
    fn naive_pdist<T: Element + Float + FromPrimitive>(
        x: &[T],
        n: usize,
        d: usize,
        metric: DistanceMetric,
    ) -> Vec<T> {
        let mut out = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                out.push(pair_distance(
                    &x[i * d..(i + 1) * d],
                    &x[j * d..(j + 1) * d],
                    metric,
                ));
            }
        }
        out
    }

    /// Runs cdist through `cdist_block_kernel` with column blocks of `block`.
    fn cdist_by_blocks<T: Element + Float + FromPrimitive>(
        x: &[T],
        y: &[T],
        (n, m, d): (usize, usize, usize),
        block: usize,
        metric: DistanceMetric,
    ) -> Vec<T> {
        let mut out = vec![<T as Element>::zero(); n * m];
        for i in 0..n {
            let mut start = 0;
            while start < m {
                let end = (start + block).min(m);
                unsafe {
                    cdist_block_kernel(
                        x.as_ptr(),
                        y.as_ptr(),
                        out.as_mut_ptr(),
                        i,
                        start,
                        end,
                        m,
                        d,
                        metric,
                    );
                }
                start = end;
            }
        }
        out
    }

    fn assert_cdist_blocks_match<T: Element + Float + FromPrimitive>() {
        for &(n, m, d) in &[(1, 37, 19), (5, 23, 8), (7, 1, 33), (3, 64, 3)] {
            let x = values::<T>(n * d, 1);
            let y = values::<T>(m * d, 2);
            for metric in METRICS {
                let expected = naive_cdist(&x, &y, (n, m, d), metric);
                let mut serial = vec![<T as Element>::zero(); n * m];
                serial_cdist(&x, &y, &mut serial, n, m, d, metric);
                let label = format!("{metric:?} n={n} m={m} d={d} serial");
                assert!(same_bits(&serial, &expected), "{label}");
                for block in [1, 4, 5, m] {
                    let blocked = cdist_by_blocks(&x, &y, (n, m, d), block, metric);
                    let label = format!("{metric:?} n={n} m={m} d={d} block={block}");
                    assert!(same_bits(&blocked, &expected), "{label}");
                }
            }
        }
    }

    fn assert_pdist_rows_match<T: Element + Float + FromPrimitive>() {
        for &(n, d) in &[(2, 5), (3, 17), (11, 9), (16, 4)] {
            let x = values::<T>(n * d, 3);
            let len = n * (n - 1) / 2;
            for metric in METRICS {
                let expected = naive_pdist(&x, n, d, metric);
                let mut serial = vec![<T as Element>::zero(); len];
                serial_pdist(&x, &mut serial, n, d, metric);
                assert!(
                    same_bits(&serial, &expected),
                    "{metric:?} n={n} d={d} serial"
                );
                // Rows in reverse order still land at the same offsets.
                let mut rows = vec![<T as Element>::zero(); len];
                for row in (0..n).rev() {
                    unsafe {
                        pdist_row_kernel(x.as_ptr(), rows.as_mut_ptr(), n, d, row, metric);
                    }
                }
                assert!(same_bits(&rows, &expected), "{metric:?} n={n} d={d}");
            }
        }
    }

    #[test]
    fn cdist_blocks_reproduce_serial_cdist() {
        assert_cdist_blocks_match::<f32>();
        assert_cdist_blocks_match::<f64>();
    }

    #[test]
    fn pdist_rows_reproduce_serial_pdist() {
        assert_pdist_rows_match::<f32>();
        assert_pdist_rows_match::<f64>();
    }

    #[test]
    fn cdist_euclidean_over_two_point_sets() {
        // X = [[0, 0], [1, 1]], Y = [[1, 0], [2, 2]]
        let x = [0.0f32, 0.0, 1.0, 1.0];
        let y = [1.0f32, 0.0, 2.0, 2.0];
        let mut out = [0.0f32; 4];

        serial_cdist(&x, &y, &mut out, 2, 2, 2, DistanceMetric::Euclidean);

        // d(x0, y0) = 1, d(x0, y1) = 2*sqrt(2), d(x1, y0) = 1, d(x1, y1) = sqrt(2)
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - (8.0f32).sqrt()).abs() < 1e-6);
        assert!((out[2] - 1.0).abs() < 1e-6);
        assert!((out[3] - (2.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn pdist_euclidean_over_one_point_set() {
        // X = [[0, 0], [1, 0], [0, 1]] - 3 points in 2D
        let x = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0];
        let mut out = [0.0f32; 3]; // 3 = n*(n-1)/2 for n=3

        serial_pdist(&x, &mut out, 3, 2, DistanceMetric::Euclidean);

        // d(0,1) = 1, d(0,2) = 1, d(1,2) = sqrt(2)
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - (2.0f32).sqrt()).abs() < 1e-6);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn f16_sqeuclidean_accumulates_in_f32() {
        // Row 0 is [32.0, 0.5 x 256], row 1 is all zeros. The squared terms are
        // 1024 followed by 256 terms of 0.25. f16 steps by 1.0 across
        // [1024, 2048), so each 0.25 rounds straight back and an f16
        // accumulator freezes at 1024; f32 reaches 1024 + 64 = 1088, which f16
        // then represents exactly.
        let d = 257;
        let mut x = vec![half::f16::from_f32(0.5); 2 * d];
        x[0] = half::f16::from_f32(32.0);
        for slot in x.iter_mut().skip(d) {
            *slot = half::f16::ZERO;
        }
        let mut out = [half::f16::ZERO; 1];

        serial_pdist(&x, &mut out, 2, d, DistanceMetric::SquaredEuclidean);

        assert_eq!(out[0].to_f32(), 1088.0);
    }
}
