// ColorOps round trips for every format, matrix, range and siting, and the
// ffmpeg cross-check.

use numr::dtype::DType;
use numr::ops::{ChromaSiting, ColorMatrix, ColorOps, ColorRange, ConvLayout, YuvSpec};
use numr::runtime::Runtime;

use super::golden::{Format, golden};
use super::planes::{blank, rgb_tensor, rgb_values, upload};

const H: usize = 8;
const W: usize = 12;

fn specs() -> Vec<YuvSpec> {
    let mut out = Vec::new();
    for matrix in [ColorMatrix::Bt601, ColorMatrix::Bt709] {
        for range in [ColorRange::Limited, ColorRange::Full] {
            for siting in [ChromaSiting::Left, ChromaSiting::Center] {
                out.push(YuvSpec::new(matrix, range, siting));
            }
        }
    }
    out
}

/// `(y_off, y_mul, c_off, c_mul)` of `range` at `bits`, as integers.
fn codes(range: ColorRange, bits: u32) -> (u32, u32, u32, u32) {
    let s = 1 << (bits - 8);
    let max = (1 << bits) - 1;
    match range {
        ColorRange::Limited => (16 * s, 219 * s, 128 * s, 224 * s),
        _ => (0, max, 1 << (bits - 1), max),
    }
}

/// RGB → YCbCr → RGB on a frame of varying luma and constant chroma stays
/// within the quantisation bound.
///
/// Chroma is constant, so downsampling loses nothing. Quantisation moves
/// `Y'` by at most `0.5 / y_mul` and `Cb'`, `Cr'` by at most `0.5 / c_mul`.
/// A channel then moves by at most `0.5 / y_mul + m · 0.5 / c_mul`, with `m`
/// the largest chroma weight: `2(1 - Kr)`, `2(1 - Kb)` or
/// `|g_cb| + |g_cr|`. The bound adds `1e-5` for f32 rounding.
pub fn check_rgb_yuv_rgb<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    for format in Format::ALL {
        for spec in specs() {
            let (kr, kb) = spec.matrix.kr_kb();
            let kg = 1.0 - kr - kb;
            let (cr_r, cb_b) = (2.0 * (1.0 - kr), 2.0 * (1.0 - kb));
            let (g_cb, g_cr) = (-2.0 * kb * (1.0 - kb) / kg, -2.0 * kr * (1.0 - kr) / kg);
            let (cb0, cr0) = (0.05, -0.04);
            let mut rgb = vec![0.0f32; 3 * H * W];
            for p in 0..H * W {
                let y = 0.2 + 0.6 * ((p * 37) % 97) as f64 / 96.0;
                rgb[p] = (y + cr_r * cr0) as f32;
                rgb[H * W + p] = (y + g_cb * cb0 + g_cr * cr0) as f32;
                rgb[2 * H * W + p] = (y + cb_b * cb0) as f32;
            }
            let (_, y_mul, _, c_mul) = codes(spec.range, format.bits());
            let m = cr_r.max(cb_b).max(g_cb.abs() + g_cr.abs());
            let bound = (0.5 / f64::from(y_mul) + m * 0.5 / f64::from(c_mul) + 1e-5) as f32;
            let input = rgb_tensor::<R>(&rgb, H, W, ConvLayout::Nchw, device);
            let set = blank::<R>(format, H, W, false, device);
            client
                .rgb_to_yuv_into(&input, set.yuv(), &spec, ConvLayout::Nchw)
                .expect("rgb_to_yuv_into");
            let back = client
                .yuv_to_rgb(set.yuv(), &spec, DType::F32, ConvLayout::Nchw)
                .expect("yuv_to_rgb");
            let got = rgb_values(&back, H, W, ConvLayout::Nchw);
            for (i, (g, w)) in got.iter().zip(&rgb).enumerate() {
                assert!(
                    (g - w).abs() <= bound,
                    "{format:?} {spec:?}: element {i}: got {g}, want {w}, bound {bound}"
                );
            }
        }
    }
}

/// YCbCr → RGB → YCbCr returns the same codes for in-gamut frames of
/// constant chroma.
///
/// The decoded RGB carries f32 rounding only, so the re-encoded `Y'`, `Cb'`
/// and `Cr'` sit within `1e-5` of the source values. Times at most 1023,
/// that stays far below the half code that rounds to a neighbour.
/// Constant chroma makes upsampling and downsampling exact.
pub fn check_yuv_rgb_yuv<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    for format in Format::ALL {
        for spec in specs() {
            let bits = format.bits();
            let shift = if format == Format::P010 { 6 } else { 0 };
            let (y_off, y_mul, c_off, _) = codes(spec.range, bits);
            let s = 1 << (bits - 8);
            let store = |v: u32| (v << shift) as u16;
            let shapes = format.plane_shapes(H, W);
            let luma: Vec<u16> = (0..H * W)
                .map(|p| store(y_off + y_mul / 4 + (p as u32 * 29) % (y_mul / 2)))
                .collect();
            let (u, v) = (store(c_off + 8 * s), store(c_off - 6 * s));
            let mut data = vec![luma];
            match format {
                Format::I420 => {
                    let n = shapes[1][0] * shapes[1][1];
                    data.push(vec![u; n]);
                    data.push(vec![v; n]);
                }
                _ => {
                    let n = shapes[1][0] * shapes[1][1] / 2;
                    data.push((0..n).flat_map(|_| [u, v]).collect());
                }
            }
            let set = upload::<R>(format, H, W, &data, false, device);
            let rgb = client
                .yuv_to_rgb(set.yuv(), &spec, DType::F32, ConvLayout::Nhwc)
                .expect("yuv_to_rgb");
            let vals = rgb_values(&rgb, H, W, ConvLayout::Nhwc);
            assert!(
                vals.iter().all(|&x| (0.05..=0.95).contains(&x)),
                "{format:?} {spec:?}: test frame left the gamut"
            );
            let back = blank::<R>(format, H, W, false, device);
            client
                .rgb_to_yuv_into(&rgb, back.yuv(), &spec, ConvLayout::Nhwc)
                .expect("rgb_to_yuv_into");
            assert_eq!(back.download(), data, "{format:?} {spec:?}");
        }
    }
}

/// BT.709 limited NV12 → RGB24 stays within PSNR 40 dB of ffmpeg's
/// `scale` filter (bilinear, accurate_rnd, full_chroma_int, left chroma).
///
/// The bound covers the filter difference: swscale interpolates chroma and
/// rounds in fixed point, numr in exact sixteenths and f32.
pub fn check_ffmpeg_psnr<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let block = golden().ffmpeg.expect(
        "color_golden.bin has no ffmpeg block; rerun gen_color_golden.py with ffmpeg on PATH",
    );
    let spec = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
    let set = upload::<R>(Format::Nv12, block.h, block.w, &block.planes, false, device);
    let rgb = client
        .yuv_to_rgb(set.yuv(), &spec, DType::F32, ConvLayout::Nhwc)
        .expect("yuv_to_rgb");
    let ours = client
        .float_to_rgb24(&rgb, ConvLayout::Nhwc)
        .expect("float_to_rgb24")
        .to_bytes()
        .expect("readback");
    assert_eq!(ours.len(), block.rgb24.len());
    let mse = ours
        .iter()
        .zip(&block.rgb24)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        / ours.len() as f64;
    let psnr = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    };
    assert!(
        psnr > 40.0,
        "PSNR against ffmpeg is {psnr:.2} dB, want > 40 dB"
    );
}
