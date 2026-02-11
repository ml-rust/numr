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

fn rand_vec_f64(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i * 17 + 3) % 1000) as f64 / 1000.0)
        .collect()
}

// ---------------------------------------------------------------------------
// numr: 2D matmul
// ---------------------------------------------------------------------------

#[flux::bench(group = "matmul_2d_f32")]
fn numr_32x32(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_numr(&[32, 32], &device);
    let bm = rand_numr(&[32, 32], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

#[flux::bench(group = "matmul_2d_f32")]
fn numr_128x128(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_numr(&[128, 128], &device);
    let bm = rand_numr(&[128, 128], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

#[flux::bench(group = "matmul_2d_f32")]
fn numr_256x256(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_numr(&[256, 256], &device);
    let bm = rand_numr(&[256, 256], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

#[flux::bench(group = "matmul_2d_f32")]
fn numr_512x512(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_numr(&[512, 512], &device);
    let bm = rand_numr(&[512, 512], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

#[flux::bench(group = "matmul_2d_f32")]
fn numr_1024x1024(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_numr(&[1024, 1024], &device);
    let bm = rand_numr(&[1024, 1024], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

// ---------------------------------------------------------------------------
// numr: 2D matmul f64
// ---------------------------------------------------------------------------

#[flux::bench(group = "matmul_2d_f64")]
fn numr_f64_128x128(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_numr_f64(&[128, 128], &device);
    let bm = rand_numr_f64(&[128, 128], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

#[flux::bench(group = "matmul_2d_f64")]
fn numr_f64_512x512(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_numr_f64(&[512, 512], &device);
    let bm = rand_numr_f64(&[512, 512], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

// ---------------------------------------------------------------------------
// numr: batched matmul
// ---------------------------------------------------------------------------

#[flux::bench(group = "matmul_batched_f32")]
fn numr_batch8_64x64(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_numr(&[8, 64, 64], &device);
    let bm = rand_numr(&[8, 64, 64], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

#[flux::bench(group = "matmul_batched_f32")]
fn numr_batch16_128x128(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_numr(&[16, 128, 128], &device);
    let bm = rand_numr(&[16, 128, 128], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

// ---------------------------------------------------------------------------
// numr: matmul_bias (fused)
// ---------------------------------------------------------------------------

#[flux::bench(group = "matmul_bias_f32")]
fn numr_bias_128x128(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_numr(&[128, 128], &device);
    let bm = rand_numr(&[128, 128], &device);
    let bias = rand_numr(&[128], &device);
    b.iter(|| black_box(client.matmul_bias(&a, &bm, &bias).unwrap()));
}

#[flux::bench(group = "matmul_bias_f32")]
fn numr_bias_512x512(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_numr(&[512, 512], &device);
    let bm = rand_numr(&[512, 512], &device);
    let bias = rand_numr(&[512], &device);
    b.iter(|| black_box(client.matmul_bias(&a, &bm, &bias).unwrap()));
}

// ---------------------------------------------------------------------------
// ndarray comparison
// ---------------------------------------------------------------------------

#[flux::bench(group = "matmul_2d_f32")]
fn ndarray_32x32(b: &mut Bencher) {
    let data_a = rand_vec_f32(32 * 32);
    let data_b = rand_vec_f32(32 * 32);
    let a = ndarray::Array2::from_shape_vec((32, 32), data_a).unwrap();
    let bm = ndarray::Array2::from_shape_vec((32, 32), data_b).unwrap();
    b.iter(|| black_box(a.dot(&bm)));
}

#[flux::bench(group = "matmul_2d_f32")]
fn ndarray_128x128(b: &mut Bencher) {
    let data_a = rand_vec_f32(128 * 128);
    let data_b = rand_vec_f32(128 * 128);
    let a = ndarray::Array2::from_shape_vec((128, 128), data_a).unwrap();
    let bm = ndarray::Array2::from_shape_vec((128, 128), data_b).unwrap();
    b.iter(|| black_box(a.dot(&bm)));
}

#[flux::bench(group = "matmul_2d_f32")]
fn ndarray_256x256(b: &mut Bencher) {
    let data_a = rand_vec_f32(256 * 256);
    let data_b = rand_vec_f32(256 * 256);
    let a = ndarray::Array2::from_shape_vec((256, 256), data_a).unwrap();
    let bm = ndarray::Array2::from_shape_vec((256, 256), data_b).unwrap();
    b.iter(|| black_box(a.dot(&bm)));
}

#[flux::bench(group = "matmul_2d_f32")]
fn ndarray_512x512(b: &mut Bencher) {
    let data_a = rand_vec_f32(512 * 512);
    let data_b = rand_vec_f32(512 * 512);
    let a = ndarray::Array2::from_shape_vec((512, 512), data_a).unwrap();
    let bm = ndarray::Array2::from_shape_vec((512, 512), data_b).unwrap();
    b.iter(|| black_box(a.dot(&bm)));
}

#[flux::bench(group = "matmul_2d_f32")]
fn ndarray_1024x1024(b: &mut Bencher) {
    let data_a = rand_vec_f32(1024 * 1024);
    let data_b = rand_vec_f32(1024 * 1024);
    let a = ndarray::Array2::from_shape_vec((1024, 1024), data_a).unwrap();
    let bm = ndarray::Array2::from_shape_vec((1024, 1024), data_b).unwrap();
    b.iter(|| black_box(a.dot(&bm)));
}

// ---------------------------------------------------------------------------
// nalgebra comparison
// ---------------------------------------------------------------------------

#[flux::bench(group = "matmul_2d_f32")]
fn nalgebra_32x32(b: &mut Bencher) {
    let a =
        nalgebra::DMatrix::<f32>::from_fn(32, 32, |i, j| ((i * 17 + j * 3) % 1000) as f32 / 1000.0);
    let bm =
        nalgebra::DMatrix::<f32>::from_fn(32, 32, |i, j| ((i * 13 + j * 7) % 1000) as f32 / 1000.0);
    b.iter(|| black_box(&a * &bm));
}

#[flux::bench(group = "matmul_2d_f32")]
fn nalgebra_128x128(b: &mut Bencher) {
    let a = nalgebra::DMatrix::<f32>::from_fn(128, 128, |i, j| {
        ((i * 17 + j * 3) % 1000) as f32 / 1000.0
    });
    let bm = nalgebra::DMatrix::<f32>::from_fn(128, 128, |i, j| {
        ((i * 13 + j * 7) % 1000) as f32 / 1000.0
    });
    b.iter(|| black_box(&a * &bm));
}

#[flux::bench(group = "matmul_2d_f32")]
fn nalgebra_512x512(b: &mut Bencher) {
    let a = nalgebra::DMatrix::<f32>::from_fn(512, 512, |i, j| {
        ((i * 17 + j * 3) % 1000) as f32 / 1000.0
    });
    let bm = nalgebra::DMatrix::<f32>::from_fn(512, 512, |i, j| {
        ((i * 13 + j * 7) % 1000) as f32 / 1000.0
    });
    b.iter(|| black_box(&a * &bm));
}

#[flux::bench(group = "matmul_2d_f32")]
fn nalgebra_1024x1024(b: &mut Bencher) {
    let a = nalgebra::DMatrix::<f32>::from_fn(1024, 1024, |i, j| {
        ((i * 17 + j * 3) % 1000) as f32 / 1000.0
    });
    let bm = nalgebra::DMatrix::<f32>::from_fn(1024, 1024, |i, j| {
        ((i * 13 + j * 7) % 1000) as f32 / 1000.0
    });
    b.iter(|| black_box(&a * &bm));
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
fn rand_cuda_f64(shape: &[usize], device: &CudaDevice) -> Tensor<CudaRuntime> {
    let client = CudaRuntime::default_client(device);
    client.rand(shape, DType::F64).unwrap()
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "matmul_2d_f32")]
fn cuda_512x512(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let a = rand_cuda(&[512, 512], &device);
    let bm = rand_cuda(&[512, 512], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "matmul_2d_f32")]
fn cuda_1024x1024(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let a = rand_cuda(&[1024, 1024], &device);
    let bm = rand_cuda(&[1024, 1024], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "matmul_2d_f64")]
fn cuda_f64_512x512(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let a = rand_cuda_f64(&[512, 512], &device);
    let bm = rand_cuda_f64(&[512, 512], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "matmul_batched_f32")]
fn cuda_batch8_64x64(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let a = rand_cuda(&[8, 64, 64], &device);
    let bm = rand_cuda(&[8, 64, 64], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

#[cfg(feature = "cuda")]
#[flux::bench(group = "matmul_bias_f32")]
fn cuda_bias_512x512(b: &mut Bencher) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let a = rand_cuda(&[512, 512], &device);
    let bm = rand_cuda(&[512, 512], &device);
    let bias = rand_cuda(&[512], &device);
    b.iter(|| black_box(client.matmul_bias(&a, &bm, &bias).unwrap()));
}

// ---------------------------------------------------------------------------
// Comparisons
// ---------------------------------------------------------------------------

#[flux::compare(
    id = "matmul_small",
    title = "Matmul 32x32 (numr vs ndarray vs nalgebra)",
    benchmarks = ["numr_32x32", "ndarray_32x32", "nalgebra_32x32"],
    baseline = "numr_32x32",
    metric = "mean"
)]
struct MatmulSmall;

#[flux::compare(
    id = "matmul_medium",
    title = "Matmul 128x128 (numr vs ndarray vs nalgebra)",
    benchmarks = ["numr_128x128", "ndarray_128x128", "nalgebra_128x128"],
    baseline = "numr_128x128",
    metric = "mean"
)]
struct MatmulMedium;

#[cfg(not(feature = "cuda"))]
#[flux::compare(
    id = "matmul_large",
    title = "Matmul 512x512 (numr vs ndarray vs nalgebra)",
    benchmarks = ["numr_512x512", "ndarray_512x512", "nalgebra_512x512"],
    baseline = "numr_512x512",
    metric = "mean"
)]
struct MatmulLarge;

#[cfg(feature = "cuda")]
#[flux::compare(
    id = "matmul_large",
    title = "Matmul 512x512 (numr vs ndarray vs nalgebra vs CUDA)",
    benchmarks = ["numr_512x512", "ndarray_512x512", "nalgebra_512x512", "cuda_512x512"],
    baseline = "numr_512x512",
    metric = "mean"
)]
struct MatmulLarge;

#[cfg(not(feature = "cuda"))]
#[flux::compare(
    id = "matmul_xlarge",
    title = "Matmul 1024x1024 (numr vs ndarray vs nalgebra)",
    benchmarks = ["numr_1024x1024", "ndarray_1024x1024", "nalgebra_1024x1024"],
    baseline = "numr_1024x1024",
    metric = "mean"
)]
struct MatmulXLarge;

#[cfg(feature = "cuda")]
#[flux::compare(
    id = "matmul_xlarge",
    title = "Matmul 1024x1024 (numr vs ndarray vs nalgebra vs CUDA)",
    benchmarks = ["numr_1024x1024", "ndarray_1024x1024", "nalgebra_1024x1024", "cuda_1024x1024"],
    baseline = "numr_1024x1024",
    metric = "mean"
)]
struct MatmulXLarge;

// ---------------------------------------------------------------------------
// Scaling series
// ---------------------------------------------------------------------------

#[flux::compare(id = "scale_32", title = "Matmul Scaling", benchmarks = ["numr_32x32"], group = "matmul_scaling", x = "32")]
struct Scale32;

#[flux::compare(id = "scale_128", title = "Matmul Scaling", benchmarks = ["numr_128x128"], group = "matmul_scaling", x = "128")]
struct Scale128;

#[flux::compare(id = "scale_512", title = "Matmul Scaling", benchmarks = ["numr_512x512"], group = "matmul_scaling", x = "512")]
struct Scale512;

#[flux::compare(id = "scale_1024", title = "Matmul Scaling", benchmarks = ["numr_1024x1024"], group = "matmul_scaling", x = "1024")]
struct Scale1024;

// ---------------------------------------------------------------------------
// Verifications: numr must be >= 90% of ndarray speed (ratio < 1.1)
// ---------------------------------------------------------------------------

#[flux::verify(expr = "numr_512x512 / ndarray_512x512 < 1.1", severity = "critical")]
struct VerifyMatmul512;

#[flux::verify(
    expr = "numr_1024x1024 / ndarray_1024x1024 < 1.1",
    severity = "critical"
)]
struct VerifyMatmul1024;

#[flux::synthetic(
    id = "matmul_512_ratio",
    formula = "numr_512x512 / ndarray_512x512",
    unit = "x"
)]
struct Matmul512Ratio;

#[flux::synthetic(
    id = "matmul_1024_ratio",
    formula = "numr_1024x1024 / ndarray_1024x1024",
    unit = "x"
)]
struct Matmul1024Ratio;

#[cfg(feature = "cuda")]
#[flux::synthetic(
    id = "cuda_speedup_512",
    formula = "numr_512x512 / cuda_512x512",
    unit = "x"
)]
struct CudaSpeedup512;

#[cfg(feature = "cuda")]
#[flux::synthetic(
    id = "cuda_speedup_1024",
    formula = "numr_1024x1024 / cuda_1024x1024",
    unit = "x"
)]
struct CudaSpeedup1024;

fn main() {
    fluxbench::run().unwrap();
}
