// ColorOps against the numpy references, on contiguous and pitched planes,
// in both layouts, into views, and in every float dtype.

use numr::dtype::DType;
use numr::ops::{ColorOps, ConvLayout};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::golden::{DecodeBlock, golden};
use super::planes::{assert_close, blank, rgb_shape, rgb_tensor, rgb_values, to_layout, upload};

pub const LAYOUTS: [ConvLayout; 2] = [ConvLayout::Nchw, ConvLayout::Nhwc];

/// Float tolerance against the numpy reference and across backends.
///
/// Every side evaluates the same correctly rounded f32 operations in the
/// same order, so the expected difference is 0. After the exact integer
/// chroma stage a value takes at most four rounded steps: a division, a
/// product and two sums, each on a value below 2. One step rounded the
/// other way moves the result by at most one ulp of 1, `2^-23`, so four
/// such steps stay within `4 · 2^-23`.
pub const TOL: f32 = 4.0 * f32::EPSILON;

fn label(block: &DecodeBlock, pitched: bool, layout: ConvLayout) -> String {
    let c = block.case;
    format!(
        "{:?} {:?} {}x{} pitched={pitched} {layout:?}",
        c.format, c.spec, c.h, c.w
    )
}

/// `yuv_to_rgb` matches every decode reference, on contiguous and pitched
/// planes, in both layouts.
pub fn check_decode_golden<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    for block in golden().decode {
        let c = block.case;
        for pitched in [false, true] {
            let set = upload::<R>(c.format, c.h, c.w, &block.planes, pitched, device);
            for layout in LAYOUTS {
                let out = client
                    .yuv_to_rgb(set.yuv(), &c.spec, DType::F32, layout)
                    .expect("yuv_to_rgb");
                assert_eq!(out.shape(), rgb_shape(c.h, c.w, layout));
                let got = rgb_values(&out, c.h, c.w, layout);
                assert_close(&got, &block.rgb, TOL, &label(&block, pitched, layout));
            }
        }
    }
}

/// `yuv_to_rgb_into` writes a channel slice of a 5-channel buffer and keeps
/// the other channels.
pub fn check_decode_into_slice<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let g = golden();
    let block = &g.decode[5];
    let c = block.case;
    let set = upload::<R>(c.format, c.h, c.w, &block.planes, true, device);
    for layout in LAYOUTS {
        let (shape, dim) = match layout {
            ConvLayout::Nchw => ([1, 5, c.h, c.w], 1),
            ConvLayout::Nhwc => ([1, c.h, c.w, 5], 3),
        };
        let buf = Tensor::<R>::full_scalar(&shape, DType::F32, 7.0, device).expect("buffer");
        let out = buf.narrow(dim, 1, 3).expect("channel slice");
        client
            .yuv_to_rgb_into(&out, set.yuv(), &c.spec, layout)
            .expect("yuv_to_rgb_into");
        let got = rgb_values(&out, c.h, c.w, layout);
        assert_close(&got, &block.rgb, TOL, &label(block, true, layout));
        for keep in [0, 4] {
            let rest = buf.narrow(dim, keep, 1).expect("kept channel");
            let bytes = rest.to_bytes().expect("readback");
            assert!(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|b| f32::from_le_bytes(*b) == 7.0),
                "{layout:?}: channel {keep} outside out changed"
            );
        }
    }
}

/// `rgb_to_yuv_into` matches every encode reference exactly, into
/// contiguous and pitched planes, from contiguous and permuted RGB.
pub fn check_encode_golden<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    for block in golden().encode {
        let c = block.case;
        let nchw = rgb_tensor::<R>(&block.rgb, c.h, c.w, ConvLayout::Nchw, device);
        let permuted = nchw.permute(&[0, 2, 3, 1]).expect("NHWC view");
        let inputs = [
            (
                rgb_tensor::<R>(&block.rgb, c.h, c.w, ConvLayout::Nhwc, device),
                ConvLayout::Nhwc,
            ),
            (nchw, ConvLayout::Nchw),
            (permuted, ConvLayout::Nhwc),
        ];
        for pitched in [false, true] {
            for (i, (rgb, layout)) in inputs.iter().enumerate() {
                let set = blank::<R>(c.format, c.h, c.w, pitched, device);
                client
                    .rgb_to_yuv_into(rgb, set.yuv(), &c.spec, *layout)
                    .expect("rgb_to_yuv_into");
                let tag = format!("{:?} {:?} pitched={pitched} input {i}", c.format, c.spec);
                assert_eq!(set.download(), block.planes, "{tag}");
                set.assert_padding_kept(c.h, c.w, &tag);
            }
        }
    }
}

fn bytes_f64(t: &Tensor<impl Runtime<DType = DType>>) -> Vec<f64> {
    t.to_bytes()
        .expect("readback")
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect()
}

/// F64 output holds the F32 result exactly, and F16 and BF16 round it once.
/// An F64 input of F32 values encodes like the F32 input.
pub fn check_float_dtypes<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let g = golden();
    for block in g.decode.iter().step_by(4) {
        let c = block.case;
        let set = upload::<R>(c.format, c.h, c.w, &block.planes, false, device);
        let layout = ConvLayout::Nhwc;
        let want = to_layout(
            &rgb_values(
                &client
                    .yuv_to_rgb(set.yuv(), &c.spec, DType::F32, layout)
                    .expect("F32"),
                c.h,
                c.w,
                layout,
            ),
            c.h,
            c.w,
            layout,
        );
        let f64_out = client
            .yuv_to_rgb(set.yuv(), &c.spec, DType::F64, layout)
            .expect("F64");
        let widened: Vec<f64> = want.iter().map(|&v| f64::from(v)).collect();
        assert_eq!(bytes_f64(&f64_out), widened, "{:?} F64", c.format);
        #[cfg(feature = "f16")]
        {
            let f16_out = client
                .yuv_to_rgb(set.yuv(), &c.spec, DType::F16, layout)
                .expect("F16");
            let f16_want: Vec<u8> = want
                .iter()
                .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                .collect();
            assert_eq!(f16_out.to_bytes().expect("F16 readback"), f16_want, "F16");
            let bf16_out = client
                .yuv_to_rgb(set.yuv(), &c.spec, DType::BF16, layout)
                .expect("BF16");
            let bf16_want: Vec<u8> = want
                .iter()
                .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
                .collect();
            assert_eq!(
                bf16_out.to_bytes().expect("BF16 readback"),
                bf16_want,
                "BF16"
            );
        }
    }
    for block in g.encode.iter().step_by(4) {
        let c = block.case;
        let vals = to_layout(&block.rgb, c.h, c.w, ConvLayout::Nchw);
        let wide: Vec<f64> = vals.iter().map(|&v| f64::from(v)).collect();
        let rgb = Tensor::<R>::from_slice(&wide, &rgb_shape(c.h, c.w, ConvLayout::Nchw), device)
            .expect("F64 rgb");
        let set = blank::<R>(c.format, c.h, c.w, false, device);
        client
            .rgb_to_yuv_into(&rgb, set.yuv(), &c.spec, ConvLayout::Nchw)
            .expect("F64 rgb_to_yuv_into");
        assert_eq!(set.download(), block.planes, "{:?} F64 encode", c.format);
        #[cfg(feature = "f16")]
        {
            let rounded: Vec<f32> = vals
                .iter()
                .map(|&v| half::f16::from_f32(v).to_f32())
                .collect();
            let narrow: Vec<half::f16> = vals.iter().map(|&v| half::f16::from_f32(v)).collect();
            let shape = rgb_shape(c.h, c.w, ConvLayout::Nchw);
            let f16_rgb = Tensor::<R>::from_slice(&narrow, &shape, device).expect("F16 rgb");
            let f32_rgb = Tensor::<R>::from_slice(&rounded, &shape, device).expect("F32 rgb");
            let (a, b) = (
                blank::<R>(c.format, c.h, c.w, false, device),
                blank::<R>(c.format, c.h, c.w, false, device),
            );
            client
                .rgb_to_yuv_into(&f16_rgb, a.yuv(), &c.spec, ConvLayout::Nchw)
                .expect("F16 encode");
            client
                .rgb_to_yuv_into(&f32_rgb, b.yuv(), &c.spec, ConvLayout::Nchw)
                .expect("F32 encode");
            assert_eq!(a.download(), b.download(), "{:?} F16 encode", c.format);
        }
    }
}
