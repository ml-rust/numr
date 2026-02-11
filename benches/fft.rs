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

fn rand_complex(n: usize, device: &CpuDevice) -> Tensor<CpuRuntime> {
    // FFT requires complex dtype — create real F64, cast to Complex128
    let client = CpuRuntime::default_client(device);
    let real = client.rand(&[n], DType::F64).unwrap();
    client.cast(&real, DType::Complex128).unwrap()
}

// ---------------------------------------------------------------------------
// numr: 1D FFT (complex, power-of-2 sizes)
// ---------------------------------------------------------------------------

#[flux::bench(group = "fft_1d_f32")]
fn numr_fft_64(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_complex(64, &device);
    b.iter(|| {
        black_box(
            client
                .fft(&t, FftDirection::Forward, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

#[flux::bench(group = "fft_1d_f32")]
fn numr_fft_256(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_complex(256, &device);
    b.iter(|| {
        black_box(
            client
                .fft(&t, FftDirection::Forward, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

#[flux::bench(group = "fft_1d_f32")]
fn numr_fft_1024(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_complex(1024, &device);
    b.iter(|| {
        black_box(
            client
                .fft(&t, FftDirection::Forward, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

#[flux::bench(group = "fft_1d_f32")]
fn numr_fft_4096(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_complex(4096, &device);
    b.iter(|| {
        black_box(
            client
                .fft(&t, FftDirection::Forward, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

#[flux::bench(group = "fft_1d_f32")]
fn numr_fft_16384(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_complex(16384, &device);
    b.iter(|| {
        black_box(
            client
                .fft(&t, FftDirection::Forward, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

#[flux::bench(group = "fft_1d_f32")]
fn numr_fft_65536(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_complex(65536, &device);
    b.iter(|| {
        black_box(
            client
                .fft(&t, FftDirection::Forward, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

// ---------------------------------------------------------------------------
// numr: real FFT (rfft)
// ---------------------------------------------------------------------------

#[flux::bench(group = "rfft_1d_f32")]
fn numr_rfft_1024(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[1024], &device);
    b.iter(|| black_box(client.rfft(&t, FftNormalization::Backward).unwrap()));
}

#[flux::bench(group = "rfft_1d_f32")]
fn numr_rfft_4096(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[4096], &device);
    b.iter(|| black_box(client.rfft(&t, FftNormalization::Backward).unwrap()));
}

#[flux::bench(group = "rfft_1d_f32")]
fn numr_rfft_65536(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[65536], &device);
    b.iter(|| black_box(client.rfft(&t, FftNormalization::Backward).unwrap()));
}

// ---------------------------------------------------------------------------
// numr: FFT round-trip (forward + inverse)
// ---------------------------------------------------------------------------

#[flux::bench(group = "fft_roundtrip_f32")]
fn numr_fft_roundtrip_1024(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_complex(1024, &device);
    b.iter(|| {
        let freq = client
            .fft(&t, FftDirection::Forward, FftNormalization::Backward)
            .unwrap();
        black_box(
            client
                .fft(&freq, FftDirection::Inverse, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

#[flux::bench(group = "fft_roundtrip_f32")]
fn numr_fft_roundtrip_16384(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_complex(16384, &device);
    b.iter(|| {
        let freq = client
            .fft(&t, FftDirection::Forward, FftNormalization::Backward)
            .unwrap();
        black_box(
            client
                .fft(&freq, FftDirection::Inverse, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

// ---------------------------------------------------------------------------
// numr: batched FFT (2D input, FFT along last dim)
// ---------------------------------------------------------------------------

#[flux::bench(group = "fft_batched_f32")]
fn numr_fft_batch32_1024(b: &mut Bencher) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_complex(32 * 1024, &device);
    // Reshape to [32, 1024] and FFT along dim -1
    let t = t.reshape(&[32, 1024]).unwrap();
    b.iter(|| {
        black_box(
            client
                .fft_dim(&t, -1, FftDirection::Forward, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

// ---------------------------------------------------------------------------
// Scaling series
// ---------------------------------------------------------------------------

#[flux::compare(id = "fscale_64", title = "FFT Scaling", benchmarks = ["numr_fft_64"], group = "fft_scaling", x = "64")]
struct FScale64;

#[flux::compare(id = "fscale_256", title = "FFT Scaling", benchmarks = ["numr_fft_256"], group = "fft_scaling", x = "256")]
struct FScale256;

#[flux::compare(id = "fscale_1024", title = "FFT Scaling", benchmarks = ["numr_fft_1024"], group = "fft_scaling", x = "1024")]
struct FScale1024;

#[flux::compare(id = "fscale_4096", title = "FFT Scaling", benchmarks = ["numr_fft_4096"], group = "fft_scaling", x = "4096")]
struct FScale4096;

#[flux::compare(id = "fscale_16384", title = "FFT Scaling", benchmarks = ["numr_fft_16384"], group = "fft_scaling", x = "16384")]
struct FScale16384;

#[flux::compare(id = "fscale_65536", title = "FFT Scaling", benchmarks = ["numr_fft_65536"], group = "fft_scaling", x = "65536")]
struct FScale65536;

fn main() {
    fluxbench::run().unwrap();
}
