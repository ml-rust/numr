// ColorOps on packed planes against CPU on U8 and U16 planes: every output
// bit for bit, on widths whose rows end inside a word, into contiguous and
// pitched planes.
//
// Call these outside `packed::with_packed`: the CPU reference uploads U8 and
// U16 planes.

use numr::dtype::DType;
use numr::ops::{ChromaSiting, ColorMatrix, ColorOps, ColorRange, ConvLayout, YuvSpec};
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;

use super::golden::{Format, golden};
use super::packed::upload_packed;
use super::planes::{blank, rgb_tensor, upload};
use super::suite_golden::LAYOUTS;
use crate::common::create_cpu_client;

/// Deterministic xorshift samples below `bound`.
pub fn noise(seed: u32, n: usize, bound: u32) -> Vec<u32> {
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

/// Frame sizes `(H, W)`. W % 4 = 2 ends NV12 and P010 chroma rows inside a
/// word, W % 8 != 0 ends I420 chroma rows inside a word and leaves a
/// partial encode group.
const SIZES: [(usize, usize); 6] = [(2, 2), (4, 6), (2, 10), (4, 14), (6, 30), (34, 130)];

fn specs() -> [YuvSpec; 3] {
    [
        YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left),
        YuvSpec::new(ColorMatrix::Bt601, ColorRange::Full, ChromaSiting::Center),
        YuvSpec::new(ColorMatrix::Bt709, ColorRange::Full, ChromaSiting::Left),
    ]
}

/// Random stored samples for every plane; P010 gets random low bits.
fn random_planes(format: Format, h: usize, w: usize, seed: u32) -> Vec<Vec<u16>> {
    let bound = if format == Format::P010 {
        1 << 16
    } else {
        1 << 8
    };
    format
        .plane_shapes(h, w)
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

/// `[3, H, W]` RGB in `[-0.1, 1.1]` with NaN, infinities, signed zeros, a
/// subnormal and exact halves mixed in.
fn random_rgb(seed: u32, h: usize, w: usize) -> Vec<f32> {
    let specials = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        1e-40,
        0.5,
        1.0,
    ];
    noise(seed, 3 * h * w, 1 << 20)
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            if i % 11 == 3 {
                specials[(v as usize) % specials.len()]
            } else {
                v as f32 / (1 << 20) as f32 * 1.2 - 0.1
            }
        })
        .collect()
}

/// Packed `yuv_to_rgb` and `rgb_to_yuv_into` on `client` give the bits of
/// CPU on U8 and U16 planes, and encode keeps every byte outside the rows.
pub fn check_packed_matches_cpu<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let (cpu, cpu_device) = create_cpu_client();
    for (n, format) in Format::ALL.into_iter().enumerate() {
        for (m, &(h, w)) in SIZES.iter().enumerate() {
            let seed = 7 + 31 * n as u32 + 5 * m as u32;
            let data = random_planes(format, h, w, seed);
            let reference = upload::<CpuRuntime>(format, h, w, &data, false, &cpu_device);
            let rgb = random_rgb(seed + 1, h, w);
            let x_cpu = rgb_tensor::<CpuRuntime>(&rgb, h, w, ConvLayout::Nhwc, &cpu_device);
            let x = rgb_tensor::<R>(&rgb, h, w, ConvLayout::Nhwc, device);
            for spec in specs() {
                let want_planes = blank::<CpuRuntime>(format, h, w, false, &cpu_device);
                cpu.rgb_to_yuv_into(&x_cpu, want_planes.yuv(), &spec, ConvLayout::Nhwc)
                    .expect("CPU encode");
                let want_planes = want_planes.download();
                for pitched in [false, true] {
                    let label = format!("{format:?} {h}x{w} {spec:?} pitched={pitched}");
                    let set = upload_packed::<R>(format, h, w, &data, pitched, device);
                    for layout in LAYOUTS {
                        let want = cpu
                            .yuv_to_rgb(reference.yuv(), &spec, DType::F32, layout)
                            .expect("CPU decode");
                        let got = client
                            .yuv_to_rgb(set.yuv(), &spec, DType::F32, layout)
                            .expect("packed decode");
                        assert_eq!(
                            got.to_bytes().expect("readback"),
                            want.to_bytes().expect("readback"),
                            "{label} {layout:?} decode"
                        );
                    }
                    let zeros: Vec<Vec<u16>> =
                        data.iter().map(|plane| vec![0; plane.len()]).collect();
                    let out = upload_packed::<R>(format, h, w, &zeros, pitched, device);
                    client
                        .rgb_to_yuv_into(&x, out.yuv(), &spec, ConvLayout::Nhwc)
                        .expect("packed encode");
                    assert_eq!(out.download(), want_planes, "{label} encode");
                    out.assert_padding_kept(h, w, &label);
                }
            }
        }
    }
}

/// NV12 decode of the ffmpeg frame, then packed RGB24, gives the bytes of
/// CPU on U8 planes and U8 RGB24, which suite_roundtrip checks against
/// ffmpeg.
pub fn check_packed_ffmpeg_bytes<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let (cpu, cpu_device) = create_cpu_client();
    let block = golden().ffmpeg.expect(
        "color_golden.bin has no ffmpeg block; rerun gen_color_golden.py with ffmpeg on PATH",
    );
    let (h, w) = (block.h, block.w);
    let spec = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
    let reference = upload::<CpuRuntime>(Format::Nv12, h, w, &block.planes, false, &cpu_device);
    let rgb = cpu
        .yuv_to_rgb(reference.yuv(), &spec, DType::F32, ConvLayout::Nhwc)
        .expect("CPU decode");
    let want = cpu
        .float_to_rgb24(&rgb, ConvLayout::Nhwc)
        .expect("CPU pack")
        .to_bytes()
        .expect("readback");
    let set = upload_packed::<R>(Format::Nv12, h, w, &block.planes, true, device);
    let rgb = client
        .yuv_to_rgb(set.yuv(), &spec, DType::F32, ConvLayout::Nhwc)
        .expect("packed decode");
    let packed = client
        .float_to_packed_rgb24(&rgb, ConvLayout::Nhwc)
        .expect("packed RGB24");
    assert_eq!(super::packed::rgb24_bytes(&packed, w), want, "ffmpeg frame");
}
