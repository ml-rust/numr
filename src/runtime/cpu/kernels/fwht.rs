//! Fast Walsh-Hadamard transform kernel.

use crate::dtype::Element;
use std::ops::{Add, Mul, Sub};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Skip Rayon below this many rows (overhead exceeds benefit).
#[cfg(feature = "rayon")]
const PARALLEL_ROW_THRESHOLD: usize = 64;

/// Sylvester-ordered in-place butterfly, unnormalized.
///
/// `block.len()` is the transform width; it need not be a power of two here,
/// callers already validated that. The inner `j` loop has no branches, so it
/// autovectorizes.
#[inline]
fn butterfly<X: Copy + Add<Output = X> + Sub<Output = X>>(block: &mut [X]) {
    let n = block.len();
    let mut len = 1;
    while len < n {
        let step = 2 * len;
        let mut i = 0;
        while i < n {
            for j in 0..len {
                let u = block[i + j];
                let v = block[i + len + j];
                block[i + j] = u + v;
                block[i + len + j] = u - v;
            }
            i += step;
        }
        len *= 2;
    }
}

/// Multiplies every element of `block` by `scale`.
#[inline]
fn scale_block<X: Copy + Mul<Output = X>>(block: &mut [X], scale: X) {
    for x in block.iter_mut() {
        *x = *x * scale;
    }
}

/// Runs the normalized transform on `row` in the element type directly.
///
/// Correct for F32 and F64, whose accumulator width already matches the
/// storage width.
#[inline]
fn fwht_row_direct<T: Element>(row: &mut [T], block_size: usize) {
    let scale = T::from_f64(1.0 / (block_size as f64).sqrt());
    for block in row.chunks_mut(block_size) {
        butterfly(block);
        scale_block(block, scale);
    }
}

/// Runs the normalized transform on `row` via an F32 accumulator, narrowing
/// back to `T` at the end.
///
/// F16 and BF16 lose the transform's cross-term cancellation if the running
/// sums stay in their own (10- or 7-bit mantissa) width, the same reason
/// `cumsum` widens narrow floats.
///
/// `scratch` is the F32 widening buffer; callers reuse one `scratch` across
/// every row they process so widening a row never allocates.
#[inline]
fn fwht_row_f32<T: Element>(row: &mut [T], block_size: usize, scratch: &mut Vec<f32>) {
    scratch.clear();
    scratch.extend(row.iter().map(|v| v.to_f32()));
    let scale = 1.0f32 / (block_size as f32).sqrt();
    for block in scratch.chunks_mut(block_size) {
        butterfly(block);
        scale_block(block, scale);
    }
    for (dst, &v) in row.iter_mut().zip(scratch.iter()) {
        *dst = T::from_f32(v);
    }
}

/// Copies `in_row` into `out_row`, multiplying by `signs` when given, then
/// runs the normalized transform on `out_row` in place.
///
/// `scratch` is passed through to [`fwht_row_f32`]; unused on the direct path.
#[inline]
fn fwht_row<T: Element>(
    in_row: &[T],
    out_row: &mut [T],
    signs: Option<&[T]>,
    block_size: usize,
    scratch: &mut Vec<f32>,
) {
    match signs {
        Some(s) => {
            for (o, (&i, &sign)) in out_row.iter_mut().zip(in_row.iter().zip(s.iter())) {
                *o = i * sign;
            }
        }
        None => out_row.copy_from_slice(in_row),
    }

    if T::DTYPE.is_narrow_float() {
        fwht_row_f32(out_row, block_size, scratch);
    } else {
        fwht_row_direct(out_row, block_size);
    }
}

/// Normalized Walsh-Hadamard transform on every `block_size` segment of every
/// row.
///
/// # Arguments
/// * `a` - Input pointer (`rows * row_width` elements, contiguous)
/// * `out` - Output pointer (`rows * row_width` elements)
/// * `signs` - Optional pointer to `row_width` sign elements, multiplied into
///   every row before the transform
/// * `row_width` - Last-dim size (a multiple of `block_size`)
/// * `rows` - Number of independent rows
/// * `block_size` - Transform width, a power of two
///
/// # Safety
/// - `a` and `out` must each point to `rows * row_width` valid, initialized,
///   properly aligned elements of type `T`, and the two must not overlap
/// - `signs`, if `Some`, must point to `row_width` valid, initialized,
///   properly aligned elements
#[inline]
pub unsafe fn fwht_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    signs: Option<*const T>,
    row_width: usize,
    rows: usize,
    block_size: usize,
) {
    use std::slice;

    let total = rows * row_width;
    let a_slice = slice::from_raw_parts(a, total);
    let out_slice = slice::from_raw_parts_mut(out, total);
    let signs_slice = signs.map(|s| slice::from_raw_parts(s, row_width));

    #[cfg(feature = "rayon")]
    if rows >= PARALLEL_ROW_THRESHOLD {
        out_slice
            .par_chunks_mut(row_width)
            .zip(a_slice.par_chunks(row_width))
            .for_each_init(Vec::new, |scratch, (out_row, in_row)| {
                fwht_row(in_row, out_row, signs_slice, block_size, scratch);
            });
        return;
    }

    let mut scratch = Vec::new();
    for (out_row, in_row) in out_slice
        .chunks_mut(row_width)
        .zip(a_slice.chunks(row_width))
    {
        fwht_row(in_row, out_row, signs_slice, block_size, &mut scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::FwhtOps;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    fn client() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (client, device)
    }

    #[test]
    fn block_size_one_is_identity() {
        let a = [1.0f32, -2.5, 3.0, 42.0];
        let mut out = [0.0f32; 4];
        unsafe {
            fwht_kernel(a.as_ptr(), out.as_mut_ptr(), None, 4, 1, 1);
        }
        assert_eq!(out, a);
    }

    #[test]
    fn block_size_two_matches_closed_form() {
        let a = [3.0f32, 1.0];
        let mut out = [0.0f32; 2];
        unsafe {
            fwht_kernel(a.as_ptr(), out.as_mut_ptr(), None, 2, 1, 2);
        }
        let s = 2.0f32.sqrt();
        assert!((out[0] - (3.0 + 1.0) / s).abs() < 1e-6);
        assert!((out[1] - (3.0 - 1.0) / s).abs() < 1e-6);
    }

    /// Sylvester-Hadamard reference: `H[i][j] = (-1)^popcount(i & j) / sqrt(n)`.
    fn sylvester_hadamard(n: usize) -> Vec<Vec<f64>> {
        let scale = 1.0 / (n as f64).sqrt();
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let bit = (i & j).count_ones() % 2;
                        if bit == 0 { scale } else { -scale }
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn block_size_eight_matches_explicit_matrix() {
        let n = 8;
        let x: Vec<f64> = vec![1.0, -2.0, 3.0, 0.5, -1.5, 4.0, -0.25, 2.0];
        let h = sylvester_hadamard(n);
        let expected: Vec<f32> = (0..n)
            .map(|i| (0..n).map(|j| h[i][j] * x[j]).sum::<f64>() as f32)
            .collect();

        let a: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let mut out = vec![0.0f32; n];
        unsafe {
            fwht_kernel(a.as_ptr(), out.as_mut_ptr(), None, n, 1, n);
        }
        for (o, e) in out.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "{o} vs {e}");
        }
    }

    #[test]
    fn applying_twice_is_identity() {
        let (client, device) = client();
        let data: Vec<f32> = (0..48).map(|i| (i as f32) * 0.37 - 5.0).collect();
        let x = Tensor::<CpuRuntime>::from_slice(&data, &[3, 16], &device).unwrap();

        let once = client.fwht(&x, 8, None).unwrap();
        let twice = client.fwht(&once, 8, None).unwrap();

        let got = twice.to_vec::<f32>();
        for (&g, &e) in got.iter().zip(data.iter()) {
            assert!((g - e).abs() < 1e-5, "{g} vs {e}");
        }
    }

    #[test]
    fn signs_equals_pre_multiplying_then_transforming_without_signs() {
        let row_width = 8;
        let a = [1.0f32, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, 8.0];
        let signs = [1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let signed: Vec<f32> = a.iter().zip(signs.iter()).map(|(&v, &s)| v * s).collect();

        let mut with_signs = [0.0f32; 8];
        unsafe {
            fwht_kernel(
                a.as_ptr(),
                with_signs.as_mut_ptr(),
                Some(signs.as_ptr()),
                row_width,
                1,
                8,
            );
        }

        let mut without_signs = [0.0f32; 8];
        unsafe {
            fwht_kernel(
                signed.as_ptr(),
                without_signs.as_mut_ptr(),
                None,
                row_width,
                1,
                8,
            );
        }

        assert_eq!(with_signs, without_signs);
    }

    #[test]
    fn block_size_not_a_power_of_two_errors() {
        let (client, device) = client();
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[6], &device).unwrap();
        assert!(client.fwht(&x, 3, None).is_err());
    }

    #[test]
    fn block_size_not_dividing_last_dim_errors() {
        let (client, device) = client();
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 16], &[16], &device).unwrap();
        assert!(client.fwht(&x, 32, None).is_err());
    }

    #[test]
    fn signs_width_mismatch_errors() {
        let (client, device) = client();
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[8], &device).unwrap();
        let signs = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device).unwrap();
        assert!(client.fwht(&x, 8, Some(&signs)).is_err());
    }
}
