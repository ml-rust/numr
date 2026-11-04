// ColorOps::pad_packed_luma against a host reference of the padded and packed
// plane, bit for bit: both sample sizes, odd rows, odd columns and both,
// multi-word rows, and tight, pitched and offset source views. Bytes outside
// a source row are SENTINEL_BYTE, so a read past the width shows in the output.

use numr::dtype::DType;
use numr::ops::{
    ChromaSiting, ColorMatrix, ColorOps, ColorRange, ConvLayout, PackedPlane, YuvPlanes, YuvSpec,
};
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

use super::packed::{PITCH_PAD_WORDS, SENTINEL_BYTE, pack_rows};
use super::planes::sample_bytes;
use super::suite_packed::noise;
use super::suite_packed_errors::invalid;
use crate::common::create_cpu_client;

/// Source `(H, W)` and target `(rows, columns)`.
const CASES: [((usize, usize), (usize, usize)); 10] = [
    // Odd rows.
    ((5, 6), (6, 6)),
    // Odd columns, with the target column inside a word at U8.
    ((4, 7), (4, 8)),
    // Odd rows and columns.
    ((5, 7), (6, 8)),
    // One pixel.
    ((1, 1), (2, 2)),
    // Multi-word rows.
    ((7, 33), (8, 34)),
    ((33, 65), (34, 66)),
    ((3, 131), (4, 132)),
    // A pad that spans several rows and words.
    ((3, 5), (9, 30)),
    // An odd target ends the output row inside a word.
    ((4, 8), (5, 9)),
    // Equal sizes give a tight copy.
    ((4, 6), (4, 6)),
];

#[derive(Debug, Clone, Copy)]
enum SourceView {
    /// Fewest words per row, offset 0.
    Tight,
    /// Row stride past the row's words.
    Pitched,
    /// Nonzero start row and word, and a row stride past the row's words.
    Offset,
}

/// Random stored samples of a `h x w` plane, low bits included.
fn samples_of(sample: DType, h: usize, w: usize, seed: u32) -> Vec<u16> {
    let bound = if sample == DType::U16 {
        1 << 16
    } else {
        1 << 8
    };
    noise(seed, h * w, bound)
        .into_iter()
        .map(|v| v as u16)
        .collect()
}

/// Bytes of the padded plane as tight words, with 0 past `columns`.
fn reference(
    samples: &[u16],
    [h, w]: [usize; 2],
    [rows, columns]: [usize; 2],
    sample: DType,
) -> Vec<u8> {
    let size = sample.size_in_bytes();
    let pitch = PackedPlane::<CpuRuntime>::row_words(columns * size) * 4;
    let mut out = vec![0u8; rows * pitch];
    for r in 0..rows {
        for c in 0..columns {
            let value = samples[r.min(h - 1) * w + c.min(w - 1)];
            let at = r * pitch + c * size;
            out[at..at + size].copy_from_slice(&sample_bytes(&[value], sample));
        }
    }
    out
}

/// U32 words holding `bytes` in the rows of `view`.
fn upload_source<R: Runtime<DType = DType>>(
    bytes: &[u8],
    h: usize,
    row_bytes: usize,
    view: SourceView,
    device: &R::Device,
) -> Tensor<R> {
    let words = PackedPlane::<R>::row_words(row_bytes);
    let upload = |data: &[u8], shape: &[usize]| {
        Tensor::<R>::from_bytes(data, shape, DType::U32, device).expect("source upload")
    };
    match view {
        SourceView::Tight => upload(&pack_rows(bytes, h, row_bytes, words), &[h, words]),
        SourceView::Pitched => {
            let pitch = words + PITCH_PAD_WORDS;
            upload(&pack_rows(bytes, h, row_bytes, pitch), &[h, pitch])
                .narrow(1, 0, words)
                .expect("pitched view")
        }
        SourceView::Offset => {
            let (first_row, first_word) = (2, 1);
            let pitch = words + first_word + PITCH_PAD_WORDS;
            let all_rows = h + first_row;
            let mut all = vec![SENTINEL_BYTE; all_rows * pitch * 4];
            for r in 0..h {
                let at = (r + first_row) * pitch * 4 + first_word * 4;
                all[at..at + row_bytes].copy_from_slice(&bytes[r * row_bytes..(r + 1) * row_bytes]);
            }
            upload(&all, &[all_rows, pitch])
                .narrow(0, first_row, h)
                .and_then(|t| t.narrow(1, first_word, words))
                .expect("offset view")
        }
    }
}

/// `pad_packed_luma` on `client` gives the bytes of the host reference.
pub fn check_pad_packed_luma<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    for sample in [DType::U8, DType::U16] {
        let size = sample.size_in_bytes();
        for (n, ((h, w), (rows, columns))) in CASES.into_iter().enumerate() {
            let samples = samples_of(sample, h, w, 101 + n as u32);
            let bytes = sample_bytes(&samples, sample);
            let want = reference(&samples, [h, w], [rows, columns], sample);
            let want_words = PackedPlane::<R>::row_words(columns * size);
            for view in [SourceView::Tight, SourceView::Pitched, SourceView::Offset] {
                let label =
                    format!("{sample:?} {h}x{w} to {rows}x{columns} from a {view:?} source");
                let y = upload_source::<R>(&bytes, h, w * size, view, device);
                let got = client
                    .pad_packed_luma(PackedPlane::new(&y, w), rows, columns, sample)
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_eq!(got.dtype(), DType::U32, "{label}: dtype");
                assert_eq!(got.shape(), [rows, want_words], "{label}: shape");
                assert_eq!(got.to_bytes().expect("readback"), want, "{label}: bytes");
            }
        }
    }
}

/// Padded luma from an odd-sized frame passes the even-size check of
/// `yuv_to_rgb`, and decodes to the bits of CPU on the reference plane.
pub fn check_padded_luma_decodes<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    device: &R::Device,
) {
    let (cpu, cpu_device) = create_cpu_client();
    let spec = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
    let [h, w, rows, columns] = [5, 7, 6, 8];
    for sample in [DType::U8, DType::U16] {
        let size = sample.size_in_bytes();
        let samples = samples_of(sample, h, w, 211);
        let y_bytes = sample_bytes(&samples, sample);
        let y = upload_source::<R>(&y_bytes, h, w * size, SourceView::Pitched, device);
        let padded = client
            .pad_packed_luma(PackedPlane::new(&y, w), rows, columns, sample)
            .expect("pad");

        let uv_samples = samples_of(sample, rows / 2, columns, 223);
        let uv_bytes = sample_bytes(&uv_samples, sample);
        let uv = upload_source::<R>(
            &uv_bytes,
            rows / 2,
            columns * size,
            SourceView::Tight,
            device,
        );
        let got = client
            .yuv_to_rgb(
                packed_planes(sample, &padded, &uv, columns),
                &spec,
                DType::F32,
                ConvLayout::Nchw,
            )
            .expect("decode of padded luma");

        let cpu_y = Tensor::<CpuRuntime>::from_bytes(
            &reference(&samples, [h, w], [rows, columns], sample),
            padded.shape(),
            DType::U32,
            &cpu_device,
        )
        .expect("reference upload");
        let cpu_uv = upload_source::<CpuRuntime>(
            &uv_bytes,
            rows / 2,
            columns * size,
            SourceView::Tight,
            &cpu_device,
        );
        let want = cpu
            .yuv_to_rgb(
                packed_planes(sample, &cpu_y, &cpu_uv, columns),
                &spec,
                DType::F32,
                ConvLayout::Nchw,
            )
            .expect("CPU decode of the reference");
        assert_eq!(
            got.to_bytes().expect("readback"),
            want.to_bytes().expect("readback"),
            "{sample:?}: decode of padded luma"
        );
    }
}

fn packed_planes<'a, R: Runtime>(
    sample: DType,
    y: &'a Tensor<R>,
    uv: &'a Tensor<R>,
    width: usize,
) -> YuvPlanes<'a, R> {
    let (y, uv) = (PackedPlane::new(y, width), PackedPlane::new(uv, width));
    match sample {
        DType::U16 => YuvPlanes::P010Packed { y, uv },
        _ => YuvPlanes::Nv12Packed { y, uv },
    }
}

/// Every documented argument error of `pad_packed_luma`.
pub fn check_pad_packed_luma_errors<R: Runtime<DType = DType>, C: ColorOps<R>>(
    client: &C,
    d: &R::Device,
) {
    let tensor = |shape: &[usize], dtype| Tensor::<R>::zeros(shape, dtype, d).expect("zeros");
    let y = tensor(&[4, 2], DType::U32);
    let plane = PackedPlane::new(&y, 6);

    invalid(
        client.pad_packed_luma(plane, 4, 6, DType::I32),
        "sample",
        "DType::U8",
    );
    invalid(
        client.pad_packed_luma(plane, 3, 6, DType::U8),
        "rows",
        "below",
    );
    invalid(
        client.pad_packed_luma(plane, 4, 5, DType::U8),
        "columns",
        "below",
    );
    invalid(
        client.pad_packed_luma(PackedPlane::new(&y, 0), 4, 6, DType::U8),
        "y",
        "no edge",
    );
    // Words of U8 samples are too short for the same width of U16 samples.
    invalid(
        client.pad_packed_luma(plane, 4, 6, DType::U16),
        "y",
        "need 3",
    );
    let bytes = tensor(&[4, 8], DType::U8);
    invalid(
        client.pad_packed_luma(PackedPlane::new(&bytes, 6), 4, 6, DType::U8),
        "y",
        "expected U32",
    );
    let strided = tensor(&[2, 4], DType::U32)
        .transpose(0, 1)
        .expect("transpose");
    invalid(
        client.pad_packed_luma(PackedPlane::new(&strided, 2), 4, 6, DType::U8),
        "y",
        "column stride",
    );
}
