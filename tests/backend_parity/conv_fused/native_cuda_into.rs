// CUDA native `conv2d_fused` on strided views: the RRDB dense block through
// `conv2d_fused_into`, and inputs given as channel slices of a wider buffer.
//
// References are the CPU result in F64 from the same half values. See
// `native_cuda` for the tolerance.

use numr::ops::{
    Conv2dConfig, ConvActivation, ConvLayout, ConvResidual, FusedConvOps, PaddingMode,
};
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

use super::native_cuda::{assert_values_close, host_f64, native_tolerance, rounded};
use super::{LEAKY_SLOPE, RESIDUAL_SCALE, pattern};
use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::{create_cpu_client, half_dtypes};

const H: usize = 17;
const W: usize = 23;
const KERNEL: usize = 3;
const BASE: usize = 64;
const GROWTH: usize = 32;
const CONVS: usize = 4;
const CHANNELS: usize = BASE + GROWTH * CONVS;
const SENTINEL: f64 = 7.0;

/// Channels `[start, start + len)` of NHWC data with `channels` per pixel.
fn channel_slice(data: &[f64], channels: usize, start: usize, len: usize) -> Vec<f64> {
    data.chunks(channels)
        .flat_map(|pixel| pixel[start..start + len].iter().copied())
        .collect()
}

/// CPU `conv2d_fused` in F64 on NHWC `x` and OHWI `w`.
fn cpu_reference(
    x: &[f64],
    x_shape: [usize; 4],
    w: &[f64],
    w_shape: [usize; 4],
    bias: &[f64],
    activation_slope: Option<f64>,
    residual: Option<(&[f64], f64)>,
) -> Vec<f64> {
    let (cpu, device) = create_cpu_client();
    let t = |data: &[f64], shape: &[usize]| {
        Tensor::<CpuRuntime>::from_slice(data, shape, &device).unwrap()
    };
    let config = Conv2dConfig::new(ConvLayout::Nhwc).with_padding(PaddingMode::Same);
    let out_shape = [x_shape[0], x_shape[1], x_shape[2], w_shape[0]];
    let residual_t = residual.map(|(data, _)| t(data, &out_shape));
    let residual = residual_t
        .as_ref()
        .zip(residual)
        .map(|(tensor, (_, scale))| ConvResidual { tensor, scale });
    let activation = match activation_slope {
        Some(slope) => ConvActivation::LeakyRelu(slope),
        None => ConvActivation::None,
    };
    cpu.conv2d_fused(
        &t(x, &x_shape),
        &t(w, &w_shape),
        Some(&t(bias, &[w_shape[0]])),
        &config,
        activation,
        residual,
    )
    .unwrap()
    .to_vec::<f64>()
}

/// Four 3x3 convs each read channels `[0, c_in)` of one buffer and write the
/// next 32 channels of the same buffer, then a fifth reads all channels and
/// adds 0.2 times its result to channels `[0, 64)`. Every written slice
/// matches the reference, and every other channel keeps its value exactly.
#[test]
fn test_native_conv_into_dense_block() {
    for dtype in half_dtypes("cuda") {
        let (rtol, atol) = native_tolerance(dtype);
        let base = rounded(pattern(H * W * BASE, 11), dtype);
        let initial: Vec<f64> = (0..H * W * CHANNELS)
            .map(|i| {
                let (p, c) = (i / CHANNELS, i % CHANNELS);
                if c < BASE {
                    base[p * BASE + c]
                } else {
                    SENTINEL
                }
            })
            .collect();
        with_cuda_backend(|client, device| {
            let make = |data: &[f64], shape: &[usize]| {
                tensor_from_f64(data, shape, dtype, &device, &client).unwrap()
            };
            let buf = make(&initial, &[1, H, W, CHANNELS]);
            let config = Conv2dConfig::new(ConvLayout::Nhwc).with_padding(PaddingMode::Same);
            let act = ConvActivation::LeakyRelu(LEAKY_SLOPE);

            for k in 0..CONVS {
                let c_in = BASE + GROWTH * k;
                let label = format!("dense conv {k} [{dtype:?}]");
                let w_shape = [GROWTH, KERNEL, KERNEL, c_in];
                let w = rounded(pattern(w_shape.iter().product(), 20 + k), dtype);
                let bias = rounded(pattern(GROWTH, 30 + k), dtype);
                let before = host_f64(&client, &buf);

                let input = buf.narrow(-1, 0, c_in).unwrap();
                let out = buf.narrow(-1, c_in, GROWTH).unwrap();
                client
                    .conv2d_fused_into(
                        &out,
                        &input,
                        &make(&w, &w_shape),
                        Some(&make(&bias, &[GROWTH])),
                        &config,
                        act,
                        None,
                    )
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                let after = host_f64(&client, &buf);

                let expected = cpu_reference(
                    &channel_slice(&before, CHANNELS, 0, c_in),
                    [1, H, W, c_in],
                    &w,
                    w_shape,
                    &bias,
                    Some(LEAKY_SLOPE),
                    None,
                );
                let written = channel_slice(&after, CHANNELS, c_in, GROWTH);
                assert_values_close(&written, &expected, rtol, atol, &label);
                for (i, (a, b)) in after.iter().zip(&before).enumerate() {
                    let c = i % CHANNELS;
                    if c < c_in || c >= c_in + GROWTH {
                        assert_eq!(a.to_bits(), b.to_bits(), "{label}: channel {c} changed");
                    }
                }
            }

            let label = format!("dense residual conv [{dtype:?}]");
            let w_shape = [BASE, KERNEL, KERNEL, CHANNELS];
            let w = rounded(pattern(w_shape.iter().product(), 40), dtype);
            let bias = rounded(pattern(BASE, 41), dtype);
            let before = host_f64(&client, &buf);
            let out = Tensor::<CudaRuntime>::zeros(&[1, H, W, BASE], dtype, &device).unwrap();
            client
                .conv2d_fused_into(
                    &out,
                    &buf,
                    &make(&w, &w_shape),
                    Some(&make(&bias, &[BASE])),
                    &config,
                    ConvActivation::None,
                    Some(ConvResidual {
                        tensor: &buf.narrow(-1, 0, BASE).unwrap(),
                        scale: RESIDUAL_SCALE,
                    }),
                )
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let residual = channel_slice(&before, CHANNELS, 0, BASE);
            let expected = cpu_reference(
                &before,
                [1, H, W, CHANNELS],
                &w,
                w_shape,
                &bias,
                None,
                Some((&residual, RESIDUAL_SCALE)),
            );
            assert_values_close(&host_f64(&client, &out), &expected, rtol, atol, &label);
        });
    }
}

/// The input is channels `[start, start + 64)` of an 80-channel buffer. Start
/// 8 keeps 16-byte alignment (128-bit staging); start 3 breaks it
/// (one-element staging). Both match the reference.
#[test]
fn test_native_conv_channel_slice_input() {
    const WIDE: usize = 80;
    const C_IN: usize = 64;
    const C_OUT: usize = 48;
    const BATCH: usize = 2;
    for dtype in half_dtypes("cuda") {
        let (rtol, atol) = native_tolerance(dtype);
        let data = rounded(pattern(BATCH * H * W * WIDE, 60), dtype);
        let w_shape = [C_OUT, KERNEL, KERNEL, C_IN];
        let w = rounded(pattern(w_shape.iter().product(), 61), dtype);
        let bias = rounded(pattern(C_OUT, 62), dtype);
        with_cuda_backend(|client, device| {
            let make = |values: &[f64], shape: &[usize]| {
                tensor_from_f64(values, shape, dtype, &device, &client).unwrap()
            };
            let buf = make(&data, &[BATCH, H, W, WIDE]);
            let w_t = make(&w, &w_shape);
            let bias_t = make(&bias, &[C_OUT]);
            let config = Conv2dConfig::new(ConvLayout::Nhwc).with_padding(PaddingMode::Same);
            for start in [8, 3] {
                let label = format!("channel slice from {start} [{dtype:?}]");
                let input = buf.narrow(-1, start, C_IN).unwrap();
                let got = client
                    .conv2d_fused(
                        &input,
                        &w_t,
                        Some(&bias_t),
                        &config,
                        ConvActivation::LeakyRelu(LEAKY_SLOPE),
                        None,
                    )
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                let expected = cpu_reference(
                    &channel_slice(&data, WIDE, start, C_IN),
                    [BATCH, H, W, C_IN],
                    &w,
                    w_shape,
                    &bias,
                    Some(LEAKY_SLOPE),
                    None,
                );
                assert_values_close(&host_f64(&client, &got), &expected, rtol, atol, &label);
            }
        });
    }
}
