#![allow(dead_code)]

use fluxbench::{Bencher, flux};
use std::hint::black_box;

use numr::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rand_numr(shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    let client = CpuRuntime::default_client(device);
    client.rand(shape, DType::F32).unwrap()
}

fn rand_numr_f64(shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
    let client = CpuRuntime::default_client(device);
    client.rand(shape, DType::F64).unwrap()
}

fn rand_vec_f32(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 17 + 3) % 1000) as f32 / 1000.0)
        .collect()
}

// ---------------------------------------------------------------------------
// numr: single-dim sum
// ---------------------------------------------------------------------------

#[flux::bench(group = "sum_single_dim_f32")]
fn numr_sum_1k(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[1000], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
}

#[flux::bench(group = "sum_single_dim_f32")]
fn numr_sum_100k(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[100_000], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
}

#[flux::bench(group = "sum_single_dim_f32")]
fn numr_sum_1m(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[1_000_000], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
}

#[flux::bench(group = "sum_single_dim_f32")]
fn numr_sum_10m(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[10_000_000], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
}

// ---------------------------------------------------------------------------
// numr: multi-dim reduce (2D matrix, reduce rows)
// ---------------------------------------------------------------------------

#[flux::bench(group = "sum_2d_rows_f32")]
fn numr_sum_rows_256x256(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[256, 256], &device);
    b.iter(|| black_box(client.sum(&t, &[1], false).unwrap()));
}

#[flux::bench(group = "sum_2d_rows_f32")]
fn numr_sum_rows_1024x1024(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[1024, 1024], &device);
    b.iter(|| black_box(client.sum(&t, &[1], false).unwrap()));
}

// ---------------------------------------------------------------------------
// numr: multi-dim reduce (reduce ALL dims)
// ---------------------------------------------------------------------------

#[flux::bench(group = "sum_all_dims_f32")]
fn numr_sum_all_256x256(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[256, 256], &device);
    b.iter(|| black_box(client.sum(&t, &[0, 1], false).unwrap()));
}

#[flux::bench(group = "sum_all_dims_f32")]
fn numr_sum_all_1024x1024(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[1024, 1024], &device);
    b.iter(|| black_box(client.sum(&t, &[0, 1], false).unwrap()));
}

// ---------------------------------------------------------------------------
// numr: mean and max
// ---------------------------------------------------------------------------

#[flux::bench(group = "mean_f32")]
fn numr_mean_1m(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[1_000_000], &device);
    b.iter(|| black_box(client.mean(&t, &[0], false).unwrap()));
}

#[flux::bench(group = "max_f32")]
fn numr_max_1m(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[1_000_000], &device);
    b.iter(|| black_box(client.max(&t, &[0], false).unwrap()));
}

// ---------------------------------------------------------------------------
// numr: f64 reductions
// ---------------------------------------------------------------------------

#[flux::bench(group = "sum_f64")]
fn numr_sum_f64_1m(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr_f64(&[1_000_000], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
}

// ---------------------------------------------------------------------------
// CUDA benchmarks
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn rand_cuda(shape: &[usize], device: &CudaDevice) -> Tensor<CudaRuntime> {
    let client = CudaRuntime::default_client(device);
    client.rand(shape, DType::F32).unwrap()
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "sum_single_dim_f32")]
fn cuda_sum_1m(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let t = rand_cuda(&[1_000_000], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "sum_single_dim_f32")]
fn cuda_sum_10m(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let t = rand_cuda(&[10_000_000], &device);
    b.iter(|| black_box(client.sum(&t, &[0], false).unwrap()));
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "sum_2d_rows_f32")]
fn cuda_sum_rows_1024x1024(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let t = rand_cuda(&[1024, 1024], &device);
    b.iter(|| black_box(client.sum(&t, &[1], false).unwrap()));
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "mean_f32")]
fn cuda_mean_1m(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let t = rand_cuda(&[1_000_000], &device);
    b.iter(|| black_box(client.mean(&t, &[0], false).unwrap()));
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "max_f32")]
fn cuda_max_1m(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let t = rand_cuda(&[1_000_000], &device);
    b.iter(|| black_box(client.max(&t, &[0], false).unwrap()));
}

// ---------------------------------------------------------------------------
// ndarray comparison
// ---------------------------------------------------------------------------

#[flux::bench(group = "sum_single_dim_f32")]
fn ndarray_sum_1k(b: &mut Bencher) {
    let data = rand_vec_f32(1000);
    let a = ndarray::Array1::from_vec(data);
    b.iter(|| black_box(a.sum()));
}

#[flux::bench(group = "sum_single_dim_f32")]
fn ndarray_sum_100k(b: &mut Bencher) {
    let data = rand_vec_f32(100_000);
    let a = ndarray::Array1::from_vec(data);
    b.iter(|| black_box(a.sum()));
}

#[flux::bench(group = "sum_single_dim_f32")]
fn ndarray_sum_1m(b: &mut Bencher) {
    let data = rand_vec_f32(1_000_000);
    let a = ndarray::Array1::from_vec(data);
    b.iter(|| black_box(a.sum()));
}

#[flux::bench(group = "sum_single_dim_f32")]
fn ndarray_sum_10m(b: &mut Bencher) {
    let data = rand_vec_f32(10_000_000);
    let a = ndarray::Array1::from_vec(data);
    b.iter(|| black_box(a.sum()));
}

#[flux::bench(group = "sum_2d_rows_f32")]
fn ndarray_sum_rows_256x256(b: &mut Bencher) {
    let data = rand_vec_f32(256 * 256);
    let a = ndarray::Array2::from_shape_vec((256, 256), data).unwrap();
    b.iter(|| black_box(a.sum_axis(ndarray::Axis(1))));
}

#[flux::bench(group = "sum_2d_rows_f32")]
fn ndarray_sum_rows_1024x1024(b: &mut Bencher) {
    let data = rand_vec_f32(1024 * 1024);
    let a = ndarray::Array2::from_shape_vec((1024, 1024), data).unwrap();
    b.iter(|| black_box(a.sum_axis(ndarray::Axis(1))));
}

#[flux::bench(group = "mean_f32")]
fn ndarray_mean_1m(b: &mut Bencher) {
    let data = rand_vec_f32(1_000_000);
    let a = ndarray::Array1::from_vec(data);
    b.iter(|| black_box(a.mean()));
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

#[cfg(not(feature = "cuda"))]
#[flux::compare(
    id = "sum_1m",
    title = "Sum 1M elements (numr vs ndarray)",
    benchmarks = ["numr_sum_1m", "ndarray_sum_1m"],
    baseline = "numr_sum_1m",
    metric = "mean"
)]
struct Sum1M;

#[cfg(feature = "cuda")]
#[flux::compare(
    id = "sum_1m",
    title = "Sum 1M elements (numr vs ndarray vs CUDA)",
    benchmarks = ["numr_sum_1m", "ndarray_sum_1m", "cuda_sum_1m"],
    baseline = "numr_sum_1m",
    metric = "mean"
)]
struct Sum1M;

#[cfg(not(feature = "cuda"))]
#[flux::compare(
    id = "sum_10m",
    title = "Sum 10M elements (numr vs ndarray)",
    benchmarks = ["numr_sum_10m", "ndarray_sum_10m"],
    baseline = "numr_sum_10m",
    metric = "mean"
)]
struct Sum10M;

#[cfg(feature = "cuda")]
#[flux::compare(
    id = "sum_10m",
    title = "Sum 10M elements (numr vs ndarray vs CUDA)",
    benchmarks = ["numr_sum_10m", "ndarray_sum_10m", "cuda_sum_10m"],
    baseline = "numr_sum_10m",
    metric = "mean"
)]
struct Sum10M;

#[cfg(not(feature = "cuda"))]
#[flux::compare(
    id = "sum_rows_1024",
    title = "Row-sum 1024x1024 (numr vs ndarray)",
    benchmarks = ["numr_sum_rows_1024x1024", "ndarray_sum_rows_1024x1024"],
    baseline = "numr_sum_rows_1024x1024",
    metric = "mean"
)]
struct SumRows1024;

#[cfg(feature = "cuda")]
#[flux::compare(
    id = "sum_rows_1024",
    title = "Row-sum 1024x1024 (numr vs ndarray vs CUDA)",
    benchmarks = ["numr_sum_rows_1024x1024", "ndarray_sum_rows_1024x1024", "cuda_sum_rows_1024x1024"],
    baseline = "numr_sum_rows_1024x1024",
    metric = "mean"
)]
struct SumRows1024;

// ---------------------------------------------------------------------------
// Scaling series
// ---------------------------------------------------------------------------

#[flux::compare(id = "rscale_1k", title = "Reduce Scaling", benchmarks = ["numr_sum_1k"], group = "reduce_scaling", x = "1000")]
struct RScale1K;

#[flux::compare(id = "rscale_100k", title = "Reduce Scaling", benchmarks = ["numr_sum_100k"], group = "reduce_scaling", x = "100000")]
struct RScale100K;

#[flux::compare(id = "rscale_1m", title = "Reduce Scaling", benchmarks = ["numr_sum_1m"], group = "reduce_scaling", x = "1000000")]
struct RScale1M;

#[flux::compare(id = "rscale_10m", title = "Reduce Scaling", benchmarks = ["numr_sum_10m"], group = "reduce_scaling", x = "10000000")]
struct RScale10M;

// ---------------------------------------------------------------------------
// Verifications: numr must be >= 90% of ndarray speed (ratio < 1.1)
// ---------------------------------------------------------------------------

#[flux::verify(expr = "numr_sum_1m / ndarray_sum_1m < 1.1", severity = "critical")]
struct VerifySum1M;

#[flux::verify(expr = "numr_sum_10m / ndarray_sum_10m < 1.1", severity = "critical")]
struct VerifySum10M;

#[flux::verify(
    expr = "numr_sum_rows_1024x1024 / ndarray_sum_rows_1024x1024 < 1.1",
    severity = "critical"
)]
struct VerifyRows1024;

#[flux::synthetic(
    id = "sum_1m_ratio",
    formula = "numr_sum_1m / ndarray_sum_1m",
    unit = "x"
)]
struct Sum1MRatio;

#[flux::synthetic(
    id = "sum_10m_ratio",
    formula = "numr_sum_10m / ndarray_sum_10m",
    unit = "x"
)]
struct Sum10MRatio;

#[cfg(feature = "cuda")]
#[flux::synthetic(
    id = "cuda_sum_speedup_1m",
    formula = "numr_sum_1m / cuda_sum_1m",
    unit = "x"
)]
struct CudaSumSpeedup1M;

#[cfg(feature = "cuda")]
#[flux::synthetic(
    id = "cuda_sum_speedup_10m",
    formula = "numr_sum_10m / cuda_sum_10m",
    unit = "x"
)]
struct CudaSumSpeedup10M;

fn main() {
    fluxbench_cli::run().unwrap();
}
