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
    let client = CpuRuntime::default_client(device);
    let real = client.rand(&[n], DType::F64).unwrap();
    client.cast(&real, DType::Complex128).unwrap()
}

// ---------------------------------------------------------------------------
// numr: 1D FFT (complex, power-of-2 sizes, parameterized)
// ---------------------------------------------------------------------------

#[flux::bench(group = "fft_1d_f32", args = [64, 256, 1024, 4096, 16384, 65536])]
fn numr_fft(b: &mut Bencher, n: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_complex(n, &device);
    b.iter(|| {
        black_box(
            client
                .fft(&t, FftDirection::Forward, FftNormalization::Backward)
                .unwrap(),
        )
    });
}

// ---------------------------------------------------------------------------
// numr: real FFT (rfft, parameterized)
// ---------------------------------------------------------------------------

#[flux::bench(group = "rfft_1d_f32", args = [1024, 4096, 65536])]
fn numr_rfft(b: &mut Bencher, n: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_numr(&[n], &device);
    b.iter(|| black_box(client.rfft(&t, FftNormalization::Backward).unwrap()));
}

// ---------------------------------------------------------------------------
// numr: FFT round-trip (forward + inverse, parameterized)
// ---------------------------------------------------------------------------

#[flux::bench(group = "fft_roundtrip_f32", args = [1024, 16384])]
fn numr_fft_roundtrip(b: &mut Bencher, n: usize) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let t = rand_complex(n, &device);
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

#[flux::compare(id = "fscale_64", title = "FFT Scaling", benchmarks = ["numr_fft@64"], group = "fft_scaling", x = "64")]
struct FScale64;

#[flux::compare(id = "fscale_256", title = "FFT Scaling", benchmarks = ["numr_fft@256"], group = "fft_scaling", x = "256")]
struct FScale256;

#[flux::compare(id = "fscale_1024", title = "FFT Scaling", benchmarks = ["numr_fft@1024"], group = "fft_scaling", x = "1024")]
struct FScale1024;

#[flux::compare(id = "fscale_4096", title = "FFT Scaling", benchmarks = ["numr_fft@4096"], group = "fft_scaling", x = "4096")]
struct FScale4096;

#[flux::compare(id = "fscale_16384", title = "FFT Scaling", benchmarks = ["numr_fft@16384"], group = "fft_scaling", x = "16384")]
struct FScale16384;

#[flux::compare(id = "fscale_65536", title = "FFT Scaling", benchmarks = ["numr_fft@65536"], group = "fft_scaling", x = "65536")]
struct FScale65536;

fn main() {
    fluxbench::run().unwrap();
}
