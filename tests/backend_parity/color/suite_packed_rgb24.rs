// ColorOps packed RGB24 against CPU on U8 RGB24: odd widths whose pixels
// cross word boundaries, pitches past the row, word views at an offset,
// strided float views, ties and clamps.

use numr::dtype::DType;
use numr::ops::{ColorOps, ConvLayout, PackedPlane};
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

use super::golden::golden;
use super::packed::{SENTINEL_BYTE, packed_rgb24, rgb24_bytes};
use super::planes::{rgb_values, to_layout};
use super::suite_golden::LAYOUTS;
use super::suite_packed::noise;
use crate::common::create_cpu_client;

/// `(H, W)` with every `3·W mod 4`, plus a frame holding all 256 values.
const SIZES: [(usize, usize); 7] = [(1, 1), (3, 1), (2, 3), (3, 5), (1, 7), (4, 13), (16, 16)];

/// Every byte of each row past `3·w` in the packed words `t` equals `want`.
fn assert_tail(t: &Tensor<impl Runtime<DType = DType>>, w: usize, want: u8, label: &str) {
    let bytes = t.to_bytes().expect("readback");
    let pitch = t.shape()[1] * 4;
    for (r, row) in bytes.chunks(pitch).enumerate() {
        for (b, &v) in row.iter().enumerate().skip(3 * w) {
            assert_eq!(v, want, "{label}: byte {b} of row {r}");
        }
    }
}

/// Packed unpack gives the CPU F32 bits, packing returns the bytes, and
/// written words keep every byte outside the rows.
pub fn check_packed_rgb24<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let (cpu, cpu_device) = create_cpu_client();
    for (n, &(h, w)) in SIZES.iter().enumerate() {
        let bytes: Vec<u8> = if (h, w) == (16, 16) {
            (0..256usize)
                .flat_map(|p| (0..3).map(move |c| ((p + 85 * c) % 256) as u8))
                .collect()
        } else {
            noise(3 + n as u32, h * w * 3, 256)
                .into_iter()
                .map(|v| v as u8)
                .collect()
        };
        let shape = [h, w, 3];
        let cpu_src = Tensor::<CpuRuntime>::from_bytes(&bytes, &shape, DType::U8, &cpu_device)
            .expect("cpu rgb24");
        for pad in [0, 2] {
            let label = format!("{h}x{w} pad={pad}");
            let src = packed_rgb24::<R>(&bytes, h, w, pad, device);
            for layout in LAYOUTS {
                let want = cpu
                    .rgb24_to_float(&cpu_src, DType::F32, layout)
                    .expect("CPU unpack")
                    .to_bytes()
                    .expect("readback");
                let f = client
                    .packed_rgb24_to_float(PackedPlane::new(&src, w), DType::F32, layout)
                    .expect("packed unpack");
                assert_eq!(f.to_bytes().expect("readback"), want, "{label} {layout:?}");

                let fresh = client.float_to_packed_rgb24(&f, layout).expect("pack");
                assert_eq!(fresh.shape(), [h, (3 * w).div_ceil(4)], "{label}");
                assert_eq!(rgb24_bytes(&fresh, w), bytes, "{label} {layout:?} pack");
                assert_tail(&fresh, w, 0, &format!("{label} new words"));

                let blank = vec![SENTINEL_BYTE; bytes.len()];
                let target = packed_rgb24::<R>(&blank, h, w, pad, device);
                client
                    .float_to_packed_rgb24_into(PackedPlane::new(&target, w), &f, layout)
                    .expect("pack into");
                assert_eq!(rgb24_bytes(&target, w), bytes, "{label} {layout:?} into");
                assert_tail(&target, w, SENTINEL_BYTE, &label);
            }
        }
    }
}

/// Words at an offset into a wider surface, strided float views on both
/// sides, and an unpack into a channel slice.
pub fn check_packed_rgb24_views<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let (h, w): (usize, usize) = (3, 5);
    let need = (3 * w).div_ceil(4);
    let bytes: Vec<u8> = (0..h * w * 3).map(|i| (i * 37 % 256) as u8).collect();
    let src = &bytes;
    let chw: Vec<f32> = (0..3)
        .flat_map(|c| (0..h * w).map(move |p| f32::from(src[p * 3 + c]) / 255.0))
        .collect();

    // Words [1, 1 + need) of each row of a surface [h, need + 3].
    let wide: Vec<u8> = (0..h)
        .flat_map(|r| {
            let mut row = vec![SENTINEL_BYTE; (need + 3) * 4];
            row[4..4 + 3 * w].copy_from_slice(&bytes[r * 3 * w..(r + 1) * 3 * w]);
            row
        })
        .collect();
    let surface =
        Tensor::<R>::from_bytes(&wide, &[h, need + 3], DType::U32, device).expect("surface");
    let view = surface.narrow(1, 1, need).expect("word view");
    for layout in LAYOUTS {
        let (shape, dim) = match layout {
            ConvLayout::Nchw => ([1, 5, h, w], 1),
            ConvLayout::Nhwc => ([1, h, w, 5], 3),
        };
        let buf = Tensor::<R>::full_scalar(&shape, DType::F32, 7.0, device).expect("buffer");
        let out = buf.narrow(dim, 1, 3).expect("channel slice");
        client
            .packed_rgb24_to_float_into(&out, PackedPlane::new(&view, w), layout)
            .expect("unpack into slice");
        assert_eq!(
            rgb_values(&out, h, w, layout),
            chw,
            "{layout:?} unpack view"
        );

        let blank = vec![SENTINEL_BYTE; wide.len()];
        let surface2 =
            Tensor::<R>::from_bytes(&blank, &[h, need + 3], DType::U32, device).expect("surface");
        let view2 = surface2.narrow(1, 1, need).expect("word view");
        client
            .float_to_packed_rgb24_into(PackedPlane::new(&view2, w), &out, layout)
            .expect("pack from slice");
        assert_eq!(
            surface2.to_bytes().expect("readback"),
            wide,
            "{layout:?} pack view"
        );
    }

    let nchw = Tensor::<R>::from_slice(&chw, &[1, 3, h, w], device).expect("rgb");
    let permuted = nchw.permute(&[0, 2, 3, 1]).expect("NHWC view");
    let packed = client
        .float_to_packed_rgb24(&permuted, ConvLayout::Nhwc)
        .expect("pack permuted");
    assert_eq!(rgb24_bytes(&packed, w), bytes, "permuted pack");
}

/// Exact `.5` products round to the even code, and out-of-range values,
/// infinities and NaN clamp, as in suite_rgb24.
pub fn check_packed_rgb24_ties_and_clamps<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let ties = golden().ties;
    let mut xs: Vec<f32> = ties.iter().map(|&(x, _)| x).collect();
    let mut want: Vec<u8> = ties.iter().map(|&(_, w)| w).collect();
    let specials = [
        (-1.0, 0),
        (-0.0, 0),
        (f32::NAN, 0),
        (f32::INFINITY, 255),
        (f32::NEG_INFINITY, 0),
        (1.5, 255),
        (0.999, 255),
        (0.5, 128),
        (1e-40, 0),
    ];
    for (x, w) in specials {
        xs.push(x);
        want.push(w);
    }
    let n = xs.len();
    let chw: Vec<f32> = (0..3).flat_map(|_| xs.iter().copied()).collect();
    let want: Vec<u8> = want.iter().flat_map(|&w| [w, w, w]).collect();
    for layout in LAYOUTS {
        let shape = match layout {
            ConvLayout::Nchw => [1, 3, 1, n],
            ConvLayout::Nhwc => [1, 1, n, 3],
        };
        let rgb =
            Tensor::<R>::from_slice(&to_layout(&chw, 1, n, layout), &shape, device).expect("rgb");
        let out = client.float_to_packed_rgb24(&rgb, layout).expect("pack");
        assert_eq!(rgb24_bytes(&out, n), want, "{layout:?} ties and clamps");
    }
}
