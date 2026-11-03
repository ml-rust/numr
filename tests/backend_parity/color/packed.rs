// Packed U32 word form of planes and RGB24 frames for ColorOps tests.
//
// Inside `with_packed`, `planes::upload` and `planes::blank` build packed
// planes, so the golden and round-trip suites run unchanged on the packed
// form. Every byte of a word outside a row's data starts as SENTINEL_BYTE.

use std::cell::Cell;

use numr::dtype::DType;
use numr::ops::PackedPlane;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::golden::Format;
use super::planes::{PlaneSet, sample_bytes};

/// Byte every word starts as outside the data of its row.
pub const SENTINEL_BYTE: u8 = 0xA5;

/// Words a pitched surface adds past the widest packed row.
pub const PITCH_PAD_WORDS: usize = 2;

thread_local! {
    static PACKED: Cell<bool> = const { Cell::new(false) };
}

/// Run `f` with `planes::upload` and `planes::blank` building packed planes.
pub fn with_packed<T>(f: impl FnOnce() -> T) -> T {
    PACKED.with(|p| p.set(true));
    let out = f();
    PACKED.with(|p| p.set(false));
    out
}

/// Whether planes upload as packed words on this thread.
pub fn packed_form() -> bool {
    PACKED.with(Cell::get)
}

/// Little-endian bytes of U32 words `[rows, words]`: row `r` holds
/// `row_bytes` bytes of `bytes` from its first word, the rest
/// [`SENTINEL_BYTE`].
pub fn pack_rows(bytes: &[u8], rows: usize, row_bytes: usize, words: usize) -> Vec<u8> {
    let pitch = words * 4;
    let mut out = vec![SENTINEL_BYTE; rows * pitch];
    for r in 0..rows {
        out[r * pitch..r * pitch + row_bytes]
            .copy_from_slice(&bytes[r * row_bytes..(r + 1) * row_bytes]);
    }
    out
}

/// The `row_bytes` data bytes of each row of the dense bytes of a words
/// view of `shape`.
pub fn unpack_rows(words: &[u8], shape: &[usize], row_bytes: usize) -> Vec<u8> {
    let pitch = shape[1] * 4;
    (0..shape[0])
        .flat_map(|r| words[r * pitch..r * pitch + row_bytes].iter().copied())
        .collect()
}

/// Upload `data` (one sample vector per plane) as packed planes of an
/// `h x w` frame.
///
/// Each plane takes the fewest words its rows need. With `pitched`, the
/// planes stack top to bottom in one U32 surface whose pitch exceeds the
/// widest plane by [`PITCH_PAD_WORDS`].
pub fn upload_packed<R: Runtime<DType = DType>>(
    format: Format,
    h: usize,
    w: usize,
    data: &[Vec<u16>],
    pitched: bool,
    device: &R::Device,
) -> PlaneSet<R> {
    let dtype = format.dtype();
    let shapes = format.plane_shapes(h, w);
    let row_bytes: Vec<usize> = shapes
        .iter()
        .map(|s| s[1] * dtype.size_in_bytes())
        .collect();
    let need: Vec<usize> = row_bytes
        .iter()
        .map(|&b| PackedPlane::<R>::row_words(b))
        .collect();
    let widths = Some(shapes.iter().map(|s| s[1]).collect());
    if !pitched {
        let planes = shapes
            .iter()
            .zip(data)
            .zip(row_bytes.iter().zip(&need))
            .map(|(([r, _], values), (&rb, &n))| {
                let bytes = pack_rows(&sample_bytes(values, dtype), *r, rb, n);
                Tensor::<R>::from_bytes(&bytes, &[*r, n], DType::U32, device)
                    .expect("packed plane upload")
            })
            .collect();
        return PlaneSet {
            format,
            surface: None,
            planes,
            packed_widths: widths,
        };
    }
    let pitch = need.iter().copied().max().unwrap_or(0) + PITCH_PAD_WORDS;
    let rows: usize = shapes.iter().map(|s| s[0]).sum();
    let shape = PackedPlane::<R>::words_shape(rows, pitch * 4).expect("word pitch");
    let mut all = vec![SENTINEL_BYTE; rows * pitch * 4];
    let mut row0 = 0;
    for (([r, _], values), &rb) in shapes.iter().zip(data).zip(&row_bytes) {
        let bytes = sample_bytes(values, dtype);
        for i in 0..*r {
            let dst = (row0 + i) * pitch * 4;
            all[dst..dst + rb].copy_from_slice(&bytes[i * rb..(i + 1) * rb]);
        }
        row0 += r;
    }
    let surface =
        Tensor::<R>::from_bytes(&all, &shape, DType::U32, device).expect("surface upload");
    let mut row0 = 0;
    let planes = shapes
        .iter()
        .zip(&need)
        .map(|(&[r, _], &n)| {
            let view = surface
                .narrow(0, row0, r)
                .and_then(|t| t.narrow(1, 0, n))
                .expect("packed plane view");
            row0 += r;
            view
        })
        .collect();
    PlaneSet {
        format,
        surface: Some(surface),
        planes,
        packed_widths: widths,
    }
}

/// Assert that every byte of a packed set outside the plane rows' data
/// keeps [`SENTINEL_BYTE`]: the tails of partial words and the pitch words.
pub fn assert_packed_padding_kept<R: Runtime<DType = DType>>(set: &PlaneSet<R>, label: &str) {
    let Some(widths) = &set.packed_widths else {
        return;
    };
    let sample = set.format.dtype().size_in_bytes();
    let mut row0 = 0;
    for (t, &cols) in set.planes.iter().zip(widths) {
        let rows = t.shape()[0];
        let (bytes, pitch, first) = match &set.surface {
            Some(s) => (
                s.to_bytes().expect("surface readback"),
                s.shape()[1] * 4,
                row0,
            ),
            None => (t.to_bytes().expect("plane readback"), t.shape()[1] * 4, 0),
        };
        for r in first..first + rows {
            for b in cols * sample..pitch {
                assert_eq!(
                    bytes[r * pitch + b],
                    SENTINEL_BYTE,
                    "{label}: padding byte {b} of word row {r}"
                );
            }
        }
        row0 += rows;
    }
}

/// Packed RGB24 words of HWC `bytes`, `[h, ceil(3w / 4) + pad_words]`.
pub fn packed_rgb24<R: Runtime<DType = DType>>(
    bytes: &[u8],
    h: usize,
    w: usize,
    pad_words: usize,
    device: &R::Device,
) -> Tensor<R> {
    let words = PackedPlane::<R>::row_words(3 * w) + pad_words;
    let packed = pack_rows(bytes, h, 3 * w, words);
    Tensor::<R>::from_bytes(&packed, &[h, words], DType::U32, device).expect("rgb24 upload")
}

/// HWC bytes of the packed RGB24 words `t` of a frame `w` pixels wide.
pub fn rgb24_bytes<R: Runtime<DType = DType>>(t: &Tensor<R>, w: usize) -> Vec<u8> {
    unpack_rows(&t.to_bytes().expect("rgb24 readback"), t.shape(), 3 * w)
}
