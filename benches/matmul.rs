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
// numr: 2D matmul (parameterized)
// ---------------------------------------------------------------------------

#[flux::bench(group = "matmul_2d_f32", args = [32, 128, 256, 512, 1024])]
fn numr_matmul(b: &mut Bencher, size: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_numr(&[size, size], &device);
    let bm = rand_numr(&[size, size], &device);
    b.iter(|| black_box(client.matmul(&a, &bm).unwrap()));
}

// ---------------------------------------------------------------------------
// numr: 2D matmul f64 (parameterized)
// ---------------------------------------------------------------------------

#[flux::bench(group = "matmul_2d_f64", args = [128, 512])]
fn numr_matmul_f64(b: &mut Bencher, size: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_numr_f64(&[size, size], &device);
    let bm = rand_numr_f64(&[size, size], &device);
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
// numr: matmul_bias (fused, parameterized)
// ---------------------------------------------------------------------------

#[flux::bench(group = "matmul_bias_f32", args = [128, 512])]
fn numr_matmul_bias(b: &mut Bencher, size: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let a = rand_numr(&[size, size], &device);
    let bm = rand_numr(&[size, size], &device);
    let bias = rand_numr(&[size], &device);
    b.iter(|| black_box(client.matmul_bias(&a, &bm, &bias).unwrap()));
}

// ---------------------------------------------------------------------------
// ndarray comparison (parameterized)
// ---------------------------------------------------------------------------

#[flux::bench(group = "matmul_2d_f32", args = [32, 128, 256, 512, 1024])]
fn ndarray_matmul(b: &mut Bencher, size: usize) {
    let data_a = rand_vec_f32(size * size);
    let data_b = rand_vec_f32(size * size);
    let a = ndarray::Array2::from_shape_vec((size, size), data_a).unwrap();
    let bm = ndarray::Array2::from_shape_vec((size, size), data_b).unwrap();
    b.iter(|| black_box(a.dot(&bm)));
}

// ---------------------------------------------------------------------------
// nalgebra comparison (parameterized)
// ---------------------------------------------------------------------------

#[flux::bench(group = "matmul_2d_f32", args = [32, 128, 512, 1024])]
fn nalgebra_matmul(b: &mut Bencher, size: usize) {
    let a = nalgebra::DMatrix::<f32>::from_fn(size, size, |i, j| {
        ((i * 17 + j * 3) % 1000) as f32 / 1000.0
    });
    let bm = nalgebra::DMatrix::<f32>::from_fn(size, size, |i, j| {
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
#[flux::bench(group = "matmul_2d_f32", args = [512, 1024])]
fn cuda_matmul(b: &mut Bencher, size: usize) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let a = rand_cuda(&[size, size], &device);
    let bm = rand_cuda(&[size, size], &device);
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
    title = "Matmul 32×32 (numr vs ndarray vs nalgebra)",
    benchmarks = ["numr_matmul@32", "ndarray_matmul@32", "nalgebra_matmul@32"],
    baseline = "numr_matmul@32",
    metric = "mean"
)]
struct MatmulSmall;

#[flux::compare(
    id = "matmul_medium",
    title = "Matmul 128×128 (numr vs ndarray vs nalgebra)",
    benchmarks = ["numr_matmul@128", "ndarray_matmul@128", "nalgebra_matmul@128"],
    baseline = "numr_matmul@128",
    metric = "mean"
)]
struct MatmulMedium;

#[cfg(not(feature = "cuda"))]
#[flux::compare(
    id = "matmul_large",
    title = "Matmul 512×512 (numr vs ndarray vs nalgebra)",
    benchmarks = ["numr_matmul@512", "ndarray_matmul@512", "nalgebra_matmul@512"],
    baseline = "numr_matmul@512",
    metric = "mean"
)]
struct MatmulLarge;

#[cfg(feature = "cuda")]
#[flux::compare(
    id = "matmul_large",
    title = "Matmul 512×512 (numr vs ndarray vs nalgebra vs CUDA)",
    benchmarks = ["numr_matmul@512", "ndarray_matmul@512", "nalgebra_matmul@512", "cuda_matmul@512"],
    baseline = "numr_matmul@512",
    metric = "mean"
)]
struct MatmulLarge;

#[cfg(not(feature = "cuda"))]
#[flux::compare(
    id = "matmul_xlarge",
    title = "Matmul 1024×1024 (numr vs ndarray vs nalgebra)",
    benchmarks = ["numr_matmul@1024", "ndarray_matmul@1024", "nalgebra_matmul@1024"],
    baseline = "numr_matmul@1024",
    metric = "mean"
)]
struct MatmulXLarge;

#[cfg(feature = "cuda")]
#[flux::compare(
    id = "matmul_xlarge",
    title = "Matmul 1024×1024 (numr vs ndarray vs nalgebra vs CUDA)",
    benchmarks = ["numr_matmul@1024", "ndarray_matmul@1024", "nalgebra_matmul@1024", "cuda_matmul@1024"],
    baseline = "numr_matmul@1024",
    metric = "mean"
)]
struct MatmulXLarge;

// ---------------------------------------------------------------------------
// Scaling series
// ---------------------------------------------------------------------------

#[flux::compare(id = "scale_32", title = "Matmul Scaling", benchmarks = ["numr_matmul@32"], group = "matmul_scaling", x = "32")]
struct Scale32;

#[flux::compare(id = "scale_128", title = "Matmul Scaling", benchmarks = ["numr_matmul@128"], group = "matmul_scaling", x = "128")]
struct Scale128;

#[flux::compare(id = "scale_512", title = "Matmul Scaling", benchmarks = ["numr_matmul@512"], group = "matmul_scaling", x = "512")]
struct Scale512;

#[flux::compare(id = "scale_1024", title = "Matmul Scaling", benchmarks = ["numr_matmul@1024"], group = "matmul_scaling", x = "1024")]
struct Scale1024;

// ---------------------------------------------------------------------------
// Verifications: numr must be >= 90% of ndarray speed (ratio < 1.1)
// ---------------------------------------------------------------------------

#[flux::verify(
    expr = "numr_matmul@512 / ndarray_matmul@512 < 1.1",
    severity = "critical"
)]
struct VerifyMatmul512;

#[flux::verify(
    expr = "numr_matmul@1024 / ndarray_matmul@1024 < 1.1",
    severity = "critical"
)]
struct VerifyMatmul1024;

#[flux::synthetic(
    id = "matmul_512_ratio",
    formula = "numr_matmul@512 / ndarray_matmul@512",
    unit = "x"
)]
struct Matmul512Ratio;

#[flux::synthetic(
    id = "matmul_1024_ratio",
    formula = "numr_matmul@1024 / ndarray_matmul@1024",
    unit = "x"
)]
struct Matmul1024Ratio;

#[cfg(feature = "cuda")]
#[flux::synthetic(
    id = "cuda_speedup_512",
    formula = "numr_matmul@512 / cuda_matmul@512",
    unit = "x"
)]
struct CudaSpeedup512;

#[cfg(feature = "cuda")]
#[flux::synthetic(
    id = "cuda_speedup_1024",
    formula = "numr_matmul@1024 / cuda_matmul@1024",
    unit = "x"
)]
struct CudaSpeedup1024;

fn main() {
    fluxbench::run().unwrap();
}
