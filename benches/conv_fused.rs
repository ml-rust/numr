#![allow(dead_code)]
//! Fused conv2d CUDA benchmarks: `FusedConvOps::conv2d_fused` and `conv2d_fused_into`.
//!
//! Covers the super-resolution workloads that the native implicit-GEMM kernel
//! targets (`src/ops/cuda/conv_fused_native.rs`): SRVGG-style body, head and
//! tail convs, and RRDB dense-block convs. Inputs are NHWC, weights are OHWI.
//! Native routing needs F16 or BF16, NHWC, one group and a contiguous weight,
//! bias and PReLU weight. A strided weight forces the generic path.

// Every benchmark here targets a CUDA kernel on half dtypes, so without those
// features the file has no bodies and even the harness imports are unused.
#[cfg(all(feature = "cuda", feature = "f16"))]
use fluxbench::{Bencher, flux};
#[cfg(all(feature = "cuda", feature = "f16"))]
use std::hint::black_box;

#[cfg(all(feature = "cuda", feature = "f16"))]
use numr::ops::{
    Conv2dConfig, ConvActivation, ConvLayout, ConvResidual, FusedConvOps, PaddingMode,
};
#[cfg(all(feature = "cuda", feature = "f16"))]
use numr::prelude::*;

#[cfg(all(feature = "cuda", feature = "f16"))]
fn rand_cuda(shape: &[usize], dtype: DType, device: &CudaDevice) -> Tensor<CudaRuntime> {
    let client = CudaRuntime::default_client(device);
    client.rand(shape, dtype).unwrap()
}

#[cfg(all(feature = "cuda", feature = "f16"))]
fn same_config() -> Conv2dConfig {
    Conv2dConfig::new(ConvLayout::Nhwc).with_padding(PaddingMode::Same)
}

/// Weight `[c_out, 3, 3, c_in]`. With `strided`, the weight is a channel-prefix
/// view of a wider tensor: same values, not contiguous.
#[cfg(all(feature = "cuda", feature = "f16"))]
fn weight_ohwi(
    c_out: usize,
    c_in: usize,
    strided: bool,
    dtype: DType,
    device: &CudaDevice,
) -> Tensor<CudaRuntime> {
    if strided {
        let wide = rand_cuda(&[c_out, 3, 3, 2 * c_in], dtype, device);
        let view = wide.narrow(-1, 0, c_in).unwrap();
        assert!(!view.is_contiguous());
        view
    } else {
        rand_cuda(&[c_out, 3, 3, c_in], dtype, device)
    }
}

/// One `conv2d_fused` call: 3x3, stride 1, Same padding, batch 1.
/// `prelu` picks per-channel PReLU, else no activation.
#[cfg(all(feature = "cuda", feature = "f16"))]
fn run_conv_fused(
    b: &mut Bencher,
    dtype: DType,
    (h, w, c_in): (usize, usize, usize),
    c_out: usize,
    prelu: bool,
    strided_weight: bool,
) {
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let input = rand_cuda(&[1, h, w, c_in], dtype, &device);
    let weight = weight_ohwi(c_out, c_in, strided_weight, dtype, &device);
    let bias = rand_cuda(&[c_out], dtype, &device);
    let slopes = rand_cuda(&[c_out], dtype, &device);
    let config = same_config();
    b.iter(|| {
        let activation = if prelu {
            ConvActivation::Prelu(&slopes)
        } else {
            ConvActivation::None
        };
        let r = black_box(
            client
                .conv2d_fused(&input, &weight, Some(&bias), &config, activation, None)
                .unwrap(),
        );
        // Sync to get accurate wall-clock time.
        client.synchronize();
        r
    });
}

// ---------------------------------------------------------------------------
// SRVGG body, head and tail convs: F16
// ---------------------------------------------------------------------------

/// Body conv at 1080p: [1,1080,1920,64] -> 64, PReLU.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[flux::bench(group = "conv_fused_srvgg_f16")]
fn cuda_conv_fused_body_1080p_64to64_prelu_f16(b: &mut Bencher) {
    run_conv_fused(b, DType::F16, (1080, 1920, 64), 64, true, false);
}

/// Body conv at 720p: [1,720,1280,64] -> 64, PReLU.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[flux::bench(group = "conv_fused_srvgg_f16")]
fn cuda_conv_fused_body_720p_64to64_prelu_f16(b: &mut Bencher) {
    run_conv_fused(b, DType::F16, (720, 1280, 64), 64, true, false);
}

/// Head conv at 1080p: [1,1080,1920,3] -> 64, PReLU. C_in = 3 takes the
/// scalar staging path.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[flux::bench(group = "conv_fused_srvgg_f16")]
fn cuda_conv_fused_head_1080p_3to64_prelu_f16(b: &mut Bencher) {
    run_conv_fused(b, DType::F16, (1080, 1920, 3), 64, true, false);
}

/// Tail conv at 1080p: [1,1080,1920,64] -> 48, no activation.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[flux::bench(group = "conv_fused_srvgg_f16")]
fn cuda_conv_fused_tail_1080p_64to48_f16(b: &mut Bencher) {
    run_conv_fused(b, DType::F16, (1080, 1920, 64), 48, false, false);
}

// ---------------------------------------------------------------------------
// RRDB dense-block convs at 540x960: F16
// ---------------------------------------------------------------------------

/// Dense-block conv: reads channels [0, c_in) of one NHWC buffer, writes the
/// next 32-channel slice [c_in, c_in + 32) of the same buffer, LeakyReLU 0.2.
#[cfg(all(feature = "cuda", feature = "f16"))]
fn run_dense_conv(b: &mut Bencher, c_in: usize) {
    let dtype = DType::F16;
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let dense = rand_cuda(&[1, 540, 960, 192], dtype, &device);
    let input = dense.narrow(-1, 0, c_in).unwrap();
    let out = dense.narrow(-1, c_in, 32).unwrap();
    let weight = weight_ohwi(32, c_in, false, dtype, &device);
    let bias = rand_cuda(&[32], dtype, &device);
    let config = same_config();
    b.iter(|| {
        client
            .conv2d_fused_into(
                &out,
                &input,
                &weight,
                Some(&bias),
                &config,
                ConvActivation::LeakyRelu(0.2),
                None,
            )
            .unwrap();
        // Sync to get accurate wall-clock time.
        client.synchronize();
        black_box(&out);
    });
}

/// Dense conv 1: C_in 64 -> 32, LeakyReLU 0.2.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[flux::bench(group = "conv_fused_rrdb_f16")]
fn cuda_conv_fused_dense_540x960_64to32_leaky_f16(b: &mut Bencher) {
    run_dense_conv(b, 64);
}

/// Dense conv 2: C_in 96 -> 32, LeakyReLU 0.2.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[flux::bench(group = "conv_fused_rrdb_f16")]
fn cuda_conv_fused_dense_540x960_96to32_leaky_f16(b: &mut Bencher) {
    run_dense_conv(b, 96);
}

/// Dense conv 3: C_in 128 -> 32, LeakyReLU 0.2.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[flux::bench(group = "conv_fused_rrdb_f16")]
fn cuda_conv_fused_dense_540x960_128to32_leaky_f16(b: &mut Bencher) {
    run_dense_conv(b, 128);
}

/// Dense conv 4: C_in 160 -> 32, LeakyReLU 0.2.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[flux::bench(group = "conv_fused_rrdb_f16")]
fn cuda_conv_fused_dense_540x960_160to32_leaky_f16(b: &mut Bencher) {
    run_dense_conv(b, 160);
}

/// Dense conv 5: C_in 192 -> 64, no activation, residual scale 0.2 into a
/// separate output. The residual is channels [0, 64) of the dense buffer.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[flux::bench(group = "conv_fused_rrdb_f16")]
fn cuda_conv_fused_dense_540x960_192to64_residual_f16(b: &mut Bencher) {
    let dtype = DType::F16;
    let device = CudaDevice::new(0);
    let client = CudaRuntime::default_client(&device);
    let dense = rand_cuda(&[1, 540, 960, 192], dtype, &device);
    let residual = dense.narrow(-1, 0, 64).unwrap();
    let out = rand_cuda(&[1, 540, 960, 64], dtype, &device);
    let weight = weight_ohwi(64, 192, false, dtype, &device);
    let bias = rand_cuda(&[64], dtype, &device);
    let config = same_config();
    b.iter(|| {
        client
            .conv2d_fused_into(
                &out,
                &dense,
                &weight,
                Some(&bias),
                &config,
                ConvActivation::None,
                Some(ConvResidual {
                    tensor: &residual,
                    scale: 0.2,
                }),
            )
            .unwrap();
        // Sync to get accurate wall-clock time.
        client.synchronize();
        black_box(&out);
    });
}

// ---------------------------------------------------------------------------
// BF16 and generic-path comparison
// ---------------------------------------------------------------------------

/// Body conv at 1080p in BF16: [1,1080,1920,64] -> 64, PReLU.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[flux::bench(group = "conv_fused_srvgg_bf16")]
fn cuda_conv_fused_body_1080p_64to64_prelu_bf16(b: &mut Bencher) {
    run_conv_fused(b, DType::BF16, (1080, 1920, 64), 64, true, false);
}

/// Body conv at 720p on the generic path. The weight is a strided view with the
/// same values, which fails only the weight-contiguity check in `native_args`.
#[cfg(all(feature = "cuda", feature = "f16"))]
#[flux::bench(group = "conv_fused_generic_f16")]
fn cuda_conv_fused_body_720p_64to64_prelu_generic_f16(b: &mut Bencher) {
    run_conv_fused(b, DType::F16, (720, 1280, 64), 64, true, true);
}

fn main() {
    fluxbench::run().unwrap();
}
