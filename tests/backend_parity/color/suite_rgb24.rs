// ColorOps RGB24 pack and unpack: exact values, ties to even, clamps,
// strided views and float dtypes.

use numr::dtype::DType;
use numr::ops::{ColorOps, ConvLayout};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::golden::golden;
use super::planes::{rgb_values, to_layout};
use super::suite_golden::LAYOUTS;

/// U8 `[16, 16, 3]` frame in which every channel takes all 256 values.
fn every_byte() -> Vec<u8> {
    (0..256usize)
        .flat_map(|p| (0..3).map(move |c| ((p + 85 * c) % 256) as u8))
        .collect()
}

fn rgb24<R: Runtime<DType = DType>>(bytes: &[u8], h: usize, w: usize, d: &R::Device) -> Tensor<R> {
    Tensor::<R>::from_bytes(bytes, &[h, w, 3], DType::U8, d).expect("rgb24 upload")
}

/// `[3, H, W]` values of HWC bytes.
fn chw(bytes: &[u8], h: usize, w: usize) -> Vec<f32> {
    (0..3)
        .flat_map(|c| (0..h * w).map(move |p| f32::from(bytes[p * 3 + c]) / 255.0))
        .collect()
}

/// Unpack gives `f32(v) / 255` exactly, and packing it back gives `v`.
pub fn check_rgb24_round_trip<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let bytes = every_byte();
    let src = rgb24::<R>(&bytes, 16, 16, device);
    let want = chw(&bytes, 16, 16);
    for layout in LAYOUTS {
        let f = client
            .rgb24_to_float(&src, DType::F32, layout)
            .expect("unpack");
        assert_eq!(rgb_values(&f, 16, 16, layout), want, "{layout:?} unpack");
        let back = client.float_to_rgb24(&f, layout).expect("pack");
        assert_eq!(back.shape(), [16, 16, 3]);
        assert_eq!(back.to_bytes().expect("readback"), bytes, "{layout:?} pack");
    }
}

/// Exact `.5` products round to the even code, and out-of-range values,
/// infinities and NaN clamp.
pub fn check_rgb24_ties_and_clamps<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let ties = golden().ties;
    assert!(ties.len() > 200, "fixture has {} ties", ties.len());
    let mut xs: Vec<f32> = ties.iter().map(|&(x, _)| x).collect();
    let mut want: Vec<u8> = ties.iter().map(|&(_, w)| w).collect();
    for &(x, w) in &ties {
        assert_eq!((x * 255.0).fract(), 0.5, "tie input {x:e} is not a tie");
        assert_eq!(w % 2, 0, "tie {x:e} wants odd code {w}");
    }
    let specials = [
        (-1.0, 0),
        (-0.0, 0),
        (f32::NAN, 0),
        (f32::INFINITY, 255),
        (f32::NEG_INFINITY, 0),
        (1.5, 255),
        (0.999, 255),
        (0.5, 128),
    ];
    for (x, w) in specials {
        xs.push(x);
        want.push(w);
    }
    let n = xs.len();
    let chw: Vec<f32> = (0..3).flat_map(|_| xs.iter().copied()).collect();
    for layout in LAYOUTS {
        let vals = to_layout(&chw, 1, n, layout);
        let shape = match layout {
            ConvLayout::Nchw => [1, 3, 1, n],
            ConvLayout::Nhwc => [1, 1, n, 3],
        };
        let rgb = Tensor::<R>::from_slice(&vals, &shape, device).expect("rgb");
        let out = client.float_to_rgb24(&rgb, layout).expect("pack");
        let got = out.to_bytes().expect("readback");
        for (i, &w) in want.iter().enumerate() {
            for c in 0..3 {
                assert_eq!(
                    got[i * 3 + c],
                    w,
                    "{layout:?}: input {:e} channel {c}",
                    xs[i]
                );
            }
        }
    }
}

/// Strided RGB24 input unpacks like its contiguous copy, and a strided U8
/// `out` keeps the bytes outside the view.
pub fn check_rgb24_views<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let (h, w, wide) = (4, 5, 7);
    let all: Vec<u8> = (0..h * wide * 3).map(|i| (i * 37 % 256) as u8).collect();
    let buf = rgb24::<R>(&all, h, wide, device);
    let view = buf.narrow(1, 1, w).expect("rgb24 view");
    let dense: Vec<u8> = (0..h)
        .flat_map(|r| all[(r * wide + 1) * 3..(r * wide + 1 + w) * 3].to_vec())
        .collect();
    let want = chw(&dense, h, w);
    let f = client
        .rgb24_to_float(&view, DType::F32, ConvLayout::Nchw)
        .expect("unpack view");
    assert_eq!(
        rgb_values(&f, h, w, ConvLayout::Nchw),
        want,
        "strided unpack"
    );

    let pad: Vec<u8> = std::iter::repeat_n(0xA5, h * wide * 3).collect();
    let target = rgb24::<R>(&pad, h, wide, device);
    let out = target.narrow(1, 1, w).expect("out view");
    client
        .float_to_rgb24_into(&out, &f, ConvLayout::Nchw)
        .expect("pack into view");
    let got = target.to_bytes().expect("readback");
    for r in 0..h {
        for c in 0..wide {
            for ch in 0..3 {
                let at = (r * wide + c) * 3 + ch;
                let want = if (1..=w).contains(&c) { all[at] } else { 0xA5 };
                assert_eq!(got[at], want, "packed byte [{r}, {c}, {ch}]");
            }
        }
    }

    let into = Tensor::<R>::zeros(&[1, h, w, 3], DType::F32, device).expect("float out");
    client
        .rgb24_to_float_into(&into, &view, ConvLayout::Nhwc)
        .expect("unpack into");
    assert_eq!(
        rgb_values(&into, h, w, ConvLayout::Nhwc),
        want,
        "unpack into"
    );
}

/// F64 holds the F32 unpack exactly, F16 and BF16 round it once, and packing
/// from an F64 tensor of F32 values matches packing the F32 tensor.
pub fn check_rgb24_dtypes<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let bytes = every_byte();
    let src = rgb24::<R>(&bytes, 16, 16, device);
    let layout = ConvLayout::Nhwc;
    let f32_vals = to_layout(&chw(&bytes, 16, 16), 16, 16, layout);
    let f64_out = client
        .rgb24_to_float(&src, DType::F64, layout)
        .expect("F64");
    let want: Vec<u8> = f32_vals
        .iter()
        .flat_map(|&v| f64::from(v).to_le_bytes())
        .collect();
    assert_eq!(f64_out.to_bytes().expect("readback"), want, "F64 unpack");
    let packed = client.float_to_rgb24(&f64_out, layout).expect("F64 pack");
    assert_eq!(packed.to_bytes().expect("readback"), bytes, "F64 pack");
    #[cfg(feature = "f16")]
    {
        let f16_out = client
            .rgb24_to_float(&src, DType::F16, layout)
            .expect("F16");
        let want: Vec<u8> = f32_vals
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        assert_eq!(f16_out.to_bytes().expect("readback"), want, "F16 unpack");
        let packed = client.float_to_rgb24(&f16_out, layout).expect("F16 pack");
        assert_eq!(packed.to_bytes().expect("readback"), bytes, "F16 pack");
        let bf16_out = client
            .rgb24_to_float(&src, DType::BF16, layout)
            .expect("BF16");
        let want: Vec<u8> = f32_vals
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        assert_eq!(bf16_out.to_bytes().expect("readback"), want, "BF16 unpack");
    }
}
