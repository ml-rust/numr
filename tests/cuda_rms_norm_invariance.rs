//! A row's CUDA F32 `rms_norm` result does not depend on how many rows share
//! the launch, nor on which of the three kernels the launcher picks for it.
//!
//! The launcher sizes the block from the row width alone, so row `r` of a
//! batched call must be the same bits as the call on row `r` by itself. The
//! packed kernel and its scalar twin own elements in the same quads and
//! accumulate in the same order, so a row whose base pointer rules packing
//! out must still match. Every kernel is also held to a CPU reference at
//! 1e-5.
//!
//! Run with:
//!   cd numr && cargo test --release --features cuda --test cuda_rms_norm_invariance

#![cfg(feature = "cuda")]

use numr::ops::NormalizationOps;
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

const EPS: f32 = 1e-5;
/// Rows in the batched call; every row is checked against its solo call.
const BATCH: usize = 7;
/// Row widths: the decode widths (packed kernel, whole and partial quad
/// slots), a width that is not a multiple of four (scalar kernel), a width
/// under one warp, and a width past the register gate (two-pass kernel).
const WIDTHS: [usize; 6] = [5120, 4096, 101, 24, 300, 8448];

fn cuda() -> Option<(CudaClient, CudaDevice)> {
    let device = CudaDevice::new(0);
    CudaClient::new(device.clone()).ok().map(|c| (c, device))
}

/// Deterministic, non-repeating values, so a mis-strided read or a dropped
/// quad changes the result.
fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + seed * 11) % 251) as f32) * 0.008 - 1.0)
        .collect()
}

fn weight(hidden: usize) -> Vec<f32> {
    (0..hidden)
        .map(|i| 0.75 + ((i as f32) * 0.011).cos() * 0.25)
        .collect()
}

fn tensor(device: &CudaDevice, data: &[f32], shape: &[usize]) -> Tensor<CudaRuntime> {
    Tensor::<CudaRuntime>::from_slice(data, shape, device).expect("tensor")
}

/// `x * rsqrt(mean(x^2) + eps) * w` in f64 for one row.
fn reference(row: &[f32], w: &[f32]) -> Vec<f64> {
    let sum_sq: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let rms_inv = 1.0 / (sum_sq / row.len() as f64 + EPS as f64).sqrt();
    row.iter()
        .zip(w)
        .map(|(&v, &g)| v as f64 * rms_inv * g as f64)
        .collect()
}

fn assert_same_bits(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label}: element {i} differs: {g} vs {w}"
        );
    }
}

fn assert_close_to_reference(got: &[f32], want: &[f64], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let diff = (*g as f64 - w).abs();
        assert!(
            diff <= 1e-5,
            "{label}: element {i} differs: {g} vs {w} (diff={diff})"
        );
    }
}

#[test]
fn rms_norm_row_is_batch_invariant() {
    let Some((client, device)) = cuda() else {
        return;
    };
    for &hidden in &WIDTHS {
        let x_data = values(BATCH * hidden, hidden);
        let w_data = weight(hidden);
        let x = tensor(&device, &x_data, &[BATCH, hidden]);
        let w = tensor(&device, &w_data, &[hidden]);

        let batched: Vec<f32> = client.rms_norm(&x, &w, EPS).expect("batched").to_vec();
        for r in 0..BATCH {
            let row = &x_data[r * hidden..(r + 1) * hidden];
            let solo_in = tensor(&device, row, &[1, hidden]);
            let solo: Vec<f32> = client.rms_norm(&solo_in, &w, EPS).expect("solo").to_vec();
            assert_same_bits(
                &batched[r * hidden..(r + 1) * hidden],
                &solo,
                &format!("hidden={hidden} row={r} batched vs solo"),
            );
            assert_close_to_reference(
                &solo,
                &reference(row, &w_data),
                &format!("hidden={hidden} row={r} vs reference"),
            );
        }
    }
}

/// A row whose base pointer is not packed-aligned takes the scalar kernel
/// and must still produce the bits of the packed kernel.
#[test]
fn rms_norm_unaligned_row_matches_aligned() {
    let Some((client, device)) = cuda() else {
        return;
    };
    let hidden = 5120;
    let row = values(hidden, 3);
    let w_data = weight(hidden);
    let w = tensor(&device, &w_data, &[hidden]);

    let aligned_in = tensor(&device, &row, &[1, hidden]);
    let aligned: Vec<f32> = client
        .rms_norm(&aligned_in, &w, EPS)
        .expect("aligned")
        .to_vec();

    // One leading element shifts the view's base pointer by four bytes.
    let mut padded = vec![0.0f32];
    padded.extend_from_slice(&row);
    let padded_in = tensor(&device, &padded, &[1, hidden + 1]);
    let shifted_in = padded_in.narrow(1, 1, hidden).expect("narrow");
    let shifted: Vec<f32> = client
        .rms_norm(&shifted_in, &w, EPS)
        .expect("shifted")
        .to_vec();

    assert_same_bits(&shifted, &aligned, "shifted view vs aligned");
    assert_close_to_reference(&shifted, &reference(&row, &w_data), "shifted vs reference");
}
