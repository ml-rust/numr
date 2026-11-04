#![cfg(feature = "cuda")]
// ColorOps on CUDA against the numpy references, and against CPU on frames
// that span many blocks: integer outputs bit for bit, float outputs within
// the tolerance of suite_golden::TOL.

use numr::dtype::DType;
use numr::ops::{ChromaSiting, ColorMatrix, ColorOps, ColorRange, ConvLayout, YuvSpec};
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

use super::golden::Format;
use super::planes::{assert_close, blank, rgb_tensor, rgb_values, upload};
use super::suite_errors::{check_rgb24_errors, check_yuv_errors};
use super::suite_golden::{
    LAYOUTS, TOL, check_decode_golden, check_decode_into_slice, check_encode_golden,
    check_float_dtypes,
};
use super::suite_rgb24::{
    check_rgb24_dtypes, check_rgb24_round_trip, check_rgb24_ties_and_clamps, check_rgb24_views,
};
use super::suite_roundtrip::{check_ffmpeg_psnr, check_rgb_yuv_rgb, check_yuv_rgb_yuv};
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::create_cpu_client;

#[test]
fn test_color_cuda_decode_matches_golden() {
    with_cuda_backend(|client, device| {
        check_decode_golden::<CudaRuntime, _>(&client, &device);
        check_decode_into_slice::<CudaRuntime, _>(&client, &device);
    });
}

#[test]
fn test_color_cuda_encode_matches_golden() {
    with_cuda_backend(|client, device| {
        check_encode_golden::<CudaRuntime, _>(&client, &device);
    });
}

#[test]
fn test_color_cuda_float_dtypes() {
    with_cuda_backend(|client, device| {
        check_float_dtypes::<CudaRuntime, _>(&client, &device);
        check_rgb24_dtypes::<CudaRuntime, _>(&client, &device);
    });
}

#[test]
fn test_color_cuda_round_trips_and_ffmpeg() {
    with_cuda_backend(|client, device| {
        check_rgb_yuv_rgb::<CudaRuntime, _>(&client, &device);
        check_yuv_rgb_yuv::<CudaRuntime, _>(&client, &device);
        check_ffmpeg_psnr::<CudaRuntime, _>(&client, &device);
    });
}

#[test]
fn test_color_cuda_rgb24_and_errors() {
    with_cuda_backend(|client, device| {
        check_rgb24_round_trip::<CudaRuntime, _>(&client, &device);
        check_rgb24_ties_and_clamps::<CudaRuntime, _>(&client, &device);
        check_rgb24_views::<CudaRuntime, _>(&client, &device);
        check_yuv_errors::<CudaRuntime, _>(&client, &device);
        check_rgb24_errors::<CudaRuntime, _>(&client, &device);
    });
}

/// Deterministic xorshift samples below `bound`.
fn noise(seed: u32, n: usize, bound: u32) -> Vec<u32> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s % bound
        })
        .collect()
}

const H: usize = 96;
const W: usize = 130;

/// Random stored samples for every plane; P010 gets random low bits.
fn random_planes(format: Format, seed: u32) -> Vec<Vec<u16>> {
    let bound = if format == Format::P010 {
        1 << 16
    } else {
        1 << 8
    };
    format
        .plane_shapes(H, W)
        .iter()
        .enumerate()
        .map(|(i, [r, c])| {
            noise(seed + i as u32, r * c, bound)
                .into_iter()
                .map(|v| v as u16)
                .collect()
        })
        .collect()
}

fn specs() -> [YuvSpec; 5] {
    [
        YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left),
        YuvSpec::new(ColorMatrix::Bt601, ColorRange::Full, ChromaSiting::Center),
        YuvSpec::new(ColorMatrix::Bt709, ColorRange::Full, ChromaSiting::Left),
        YuvSpec::new(ColorMatrix::Fcc, ColorRange::Limited, ChromaSiting::Center),
        YuvSpec::new(ColorMatrix::Smpte240m, ColorRange::Full, ChromaSiting::Left),
    ]
}

#[test]
fn test_color_cuda_matches_cpu_on_large_frames() {
    with_cuda_backend(|cuda, cuda_device| {
        let (cpu, cpu_device) = create_cpu_client();
        for (n, format) in Format::ALL.into_iter().enumerate() {
            for spec in specs() {
                let label = format!("{format:?} {spec:?}");
                let data = random_planes(format, 7 + 3 * n as u32);
                let a = upload::<CpuRuntime>(format, H, W, &data, false, &cpu_device);
                let b = upload::<CudaRuntime>(format, H, W, &data, true, &cuda_device);
                for layout in LAYOUTS {
                    let want = cpu
                        .yuv_to_rgb(a.yuv(), &spec, DType::F32, layout)
                        .expect("CPU");
                    let got = cuda
                        .yuv_to_rgb(b.yuv(), &spec, DType::F32, layout)
                        .expect("CUDA");
                    assert_close(
                        &rgb_values(&got, H, W, layout),
                        &rgb_values(&want, H, W, layout),
                        TOL,
                        &format!("{label} {layout:?} decode"),
                    );
                }

                let rgb: Vec<f32> = noise(99 + n as u32, 3 * H * W, 1 << 20)
                    .into_iter()
                    .map(|v| v as f32 / (1 << 20) as f32 * 1.2 - 0.1)
                    .collect();
                let x_cpu = rgb_tensor::<CpuRuntime>(&rgb, H, W, ConvLayout::Nhwc, &cpu_device);
                let x_cuda = rgb_tensor::<CudaRuntime>(&rgb, H, W, ConvLayout::Nhwc, &cuda_device);
                let pa = blank::<CpuRuntime>(format, H, W, false, &cpu_device);
                let pb = blank::<CudaRuntime>(format, H, W, true, &cuda_device);
                cpu.rgb_to_yuv_into(&x_cpu, pa.yuv(), &spec, ConvLayout::Nhwc)
                    .expect("CPU encode");
                cuda.rgb_to_yuv_into(&x_cuda, pb.yuv(), &spec, ConvLayout::Nhwc)
                    .expect("CUDA encode");
                assert_eq!(pb.download(), pa.download(), "{label} encode");
                pb.assert_padding_kept(H, W, &label);
            }
        }

        let bytes: Vec<u8> = noise(5, H * W * 3, 256)
            .into_iter()
            .map(|v| v as u8)
            .collect();
        let shape = [H, W, 3];
        let src_cpu =
            Tensor::<CpuRuntime>::from_bytes(&bytes, &shape, DType::U8, &cpu_device).expect("cpu");
        let src_cuda = Tensor::<CudaRuntime>::from_bytes(&bytes, &shape, DType::U8, &cuda_device)
            .expect("gpu");
        for layout in LAYOUTS {
            let f_cpu = cpu
                .rgb24_to_float(&src_cpu, DType::F32, layout)
                .expect("CPU");
            let f_cuda = cuda
                .rgb24_to_float(&src_cuda, DType::F32, layout)
                .expect("CUDA");
            assert_eq!(
                f_cuda.to_bytes().expect("readback"),
                f_cpu.to_bytes().expect("readback"),
                "{layout:?} unpack"
            );
            let back = cuda.float_to_rgb24(&f_cuda, layout).expect("CUDA pack");
            assert_eq!(back.to_bytes().expect("readback"), bytes, "{layout:?} pack");
        }
    });
}

/// CUDA F16 and BF16 outputs round the F32 result once, so they stay within
/// one ulp at 1 of the narrow type of the CPU F32 result.
#[test]
fn test_color_cuda_half_outputs_match_cpu_f32() {
    fn narrow_values<R: Runtime<DType = DType>>(t: &Tensor<R>) -> Vec<f32> {
        let bytes = t.to_bytes().expect("readback");
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                match t.dtype() {
                    DType::BF16 => f32::from_bits(u32::from(bits) << 16),
                    _ => f16_bits_to_f32(bits),
                }
            })
            .collect()
    }
    with_cuda_backend(|cuda, cuda_device| {
        let (cpu, cpu_device) = create_cpu_client();
        let spec = specs()[0];
        for format in Format::ALL {
            let data = random_planes(format, 41);
            let a = upload::<CpuRuntime>(format, H, W, &data, false, &cpu_device);
            let b = upload::<CudaRuntime>(format, H, W, &data, false, &cuda_device);
            let layout = ConvLayout::Nchw;
            let want = cpu
                .yuv_to_rgb(a.yuv(), &spec, DType::F32, layout)
                .expect("CPU");
            let want = rgb_values(&want, H, W, layout);
            for (dtype, ulp) in [(DType::F16, 2f32.powi(-10)), (DType::BF16, 2f32.powi(-7))] {
                let got = cuda
                    .yuv_to_rgb(b.yuv(), &spec, dtype, layout)
                    .expect("CUDA");
                let label = format!("{format:?} {dtype:?}");
                assert_close(&narrow_values(&got), &want, ulp, &label);
            }
        }
    });
}

/// IEEE binary16 bits to f32.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1F);
    let frac = f32::from(bits & 0x3FF);
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 if frac == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}
