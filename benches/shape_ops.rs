#![allow(dead_code)]

use fluxbench::{Bencher, flux};
use std::hint::black_box;

use numr::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn setup() -> (CpuDevice, CpuClient) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    (device, client)
}

fn rand_t(shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    let client = CpuRuntime::default_client(device);
    client.rand(shape, DType::F32).unwrap()
}

// ---------------------------------------------------------------------------
// repeat
// ---------------------------------------------------------------------------

#[flux::bench(group = "repeat_f32")]
fn numr_repeat_256x256_2x2(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[256, 256], &device);
    b.iter(|| black_box(client.repeat(&t, &[2, 2]).unwrap()));
}

#[flux::bench(group = "repeat_f32")]
fn numr_repeat_1024x64_4x1(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[1024, 64], &device);
    b.iter(|| black_box(client.repeat(&t, &[4, 1]).unwrap()));
}

// ---------------------------------------------------------------------------
// repeat_interleave
// ---------------------------------------------------------------------------

#[flux::bench(group = "repeat_interleave_f32")]
fn numr_repeat_interleave_1k_x4(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[1000], &device);
    b.iter(|| black_box(client.repeat_interleave(&t, 4, Some(0)).unwrap()));
}

#[flux::bench(group = "repeat_interleave_f32")]
fn numr_repeat_interleave_256x64_x4(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[256, 64], &device);
    b.iter(|| black_box(client.repeat_interleave(&t, 4, Some(0)).unwrap()));
}

// ---------------------------------------------------------------------------
// unfold (sliding window)
// ---------------------------------------------------------------------------

#[flux::bench(group = "unfold_f32")]
fn numr_unfold_10k_win64_step1(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[10_000], &device);
    b.iter(|| black_box(client.unfold(&t, 0, 64, 1).unwrap()));
}

#[flux::bench(group = "unfold_f32")]
fn numr_unfold_10k_win64_step32(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[10_000], &device);
    b.iter(|| black_box(client.unfold(&t, 0, 64, 32).unwrap()));
}

#[flux::bench(group = "unfold_f32")]
fn numr_unfold_100k_win256_step128(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[100_000], &device);
    b.iter(|| black_box(client.unfold(&t, 0, 256, 128).unwrap()));
}

// ---------------------------------------------------------------------------
// cat (concatenation)
// ---------------------------------------------------------------------------

#[flux::bench(group = "cat_f32")]
fn numr_cat_10x_1000(b: &mut Bencher) {
    let (device, client) = setup();
    let tensors: Vec<_> = (0..10).map(|_| rand_t(&[1000], &device)).collect();
    let refs: Vec<&Tensor<CpuRuntime>> = tensors.iter().collect();
    b.iter(|| black_box(client.cat(&refs, 0).unwrap()));
}

#[flux::bench(group = "cat_f32")]
fn numr_cat_10x_256x64(b: &mut Bencher) {
    let (device, client) = setup();
    let tensors: Vec<_> = (0..10).map(|_| rand_t(&[256, 64], &device)).collect();
    let refs: Vec<&Tensor<CpuRuntime>> = tensors.iter().collect();
    b.iter(|| black_box(client.cat(&refs, 0).unwrap()));
}

// ---------------------------------------------------------------------------
// stack
// ---------------------------------------------------------------------------

#[flux::bench(group = "stack_f32")]
fn numr_stack_8x_1000(b: &mut Bencher) {
    let (device, client) = setup();
    let tensors: Vec<_> = (0..8).map(|_| rand_t(&[1000], &device)).collect();
    let refs: Vec<&Tensor<CpuRuntime>> = tensors.iter().collect();
    b.iter(|| black_box(client.stack(&refs, 0).unwrap()));
}

// ---------------------------------------------------------------------------
// split / chunk
// ---------------------------------------------------------------------------

#[flux::bench(group = "split_f32")]
fn numr_split_10k_into_100(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[10_000], &device);
    b.iter(|| black_box(client.split(&t, 100, 0).unwrap()));
}

#[flux::bench(group = "split_f32")]
fn numr_chunk_10k_into_10(b: &mut Bencher) {
    let (device, client) = setup();
    let t = rand_t(&[10_000], &device);
    b.iter(|| black_box(client.chunk(&t, 10, 0).unwrap()));
}

// ---------------------------------------------------------------------------
// ndarray comparison: repeat via broadcast + to_owned
// ---------------------------------------------------------------------------

#[flux::bench(group = "cat_f32")]
fn ndarray_cat_10x_1000(b: &mut Bencher) {
    let vecs: Vec<ndarray::Array1<f32>> = (0..10)
        .map(|_| ndarray::Array1::from_vec((0..1000).map(|i| (i as f32) / 1000.0).collect()))
        .collect();
    let views: Vec<ndarray::ArrayView1<f32>> = vecs.iter().map(|a| a.view()).collect();
    b.iter(|| black_box(ndarray::concatenate(ndarray::Axis(0), &views).unwrap()));
}

#[flux::bench(group = "cat_f32")]
fn ndarray_cat_10x_256x64(b: &mut Bencher) {
    let vecs: Vec<ndarray::Array2<f32>> = (0..10)
        .map(|_| {
            ndarray::Array2::from_shape_vec(
                (256, 64),
                (0..256 * 64).map(|i| (i as f32) / 16384.0).collect(),
            )
            .unwrap()
        })
        .collect();
    let views: Vec<ndarray::ArrayView2<f32>> = vecs.iter().map(|a| a.view()).collect();
    b.iter(|| black_box(ndarray::concatenate(ndarray::Axis(0), &views).unwrap()));
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

#[flux::compare(
    id = "cat_1d",
    title = "Concatenate 10x 1000-elem (numr vs ndarray)",
    benchmarks = ["numr_cat_10x_1000", "ndarray_cat_10x_1000"],
    baseline = "numr_cat_10x_1000",
    metric = "mean"
)]
struct Cat1D;

#[flux::compare(
    id = "cat_2d",
    title = "Concatenate 10x 256x64 (numr vs ndarray)",
    benchmarks = ["numr_cat_10x_256x64", "ndarray_cat_10x_256x64"],
    baseline = "numr_cat_10x_256x64",
    metric = "mean"
)]
struct Cat2D;

// ---------------------------------------------------------------------------
// Verifications: numr must be >= 90% of ndarray speed (ratio < 1.1)
// ---------------------------------------------------------------------------

#[flux::verify(
    expr = "numr_cat_10x_1000 / ndarray_cat_10x_1000 < 1.2",
    severity = "critical"
)]
struct VerifyCat1D;

#[flux::verify(
    expr = "numr_cat_10x_256x64 / ndarray_cat_10x_256x64 < 1.2",
    severity = "critical"
)]
struct VerifyCat2D;

#[flux::synthetic(
    id = "cat_1d_ratio",
    formula = "numr_cat_10x_1000 / ndarray_cat_10x_1000",
    unit = "x"
)]
struct Cat1DRatio;

#[flux::synthetic(
    id = "cat_2d_ratio",
    formula = "numr_cat_10x_256x64 / ndarray_cat_10x_256x64",
    unit = "x"
)]
struct Cat2DRatio;

fn main() {
    fluxbench_cli::run().unwrap();
}
