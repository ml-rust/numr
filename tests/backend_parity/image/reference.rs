// Host references for ImageOps, computed from the index formulas on raw
// element bytes.
//
// Every op moves elements without computing, so a reference that gathers
// element-sized byte chunks covers every dtype. Results must match exactly.

use numr::dtype::DType;
use numr::ops::ConvLayout;

/// Source map of one reference op.
#[derive(Debug, Clone)]
pub enum RefOp {
    /// Pixel shuffle with factor `r`.
    Shuffle(usize),
    /// Pixel unshuffle with factor `r`.
    Unshuffle(usize),
    /// Nearest upsampling with explicit source index tables for H and W.
    Nearest {
        /// Source H index per output H index.
        rows: Vec<usize>,
        /// Source W index per output W index.
        cols: Vec<usize>,
    },
}

/// Canonical `[N, C, H, W]` extents of a shape in `layout` order.
pub fn to_canonical(shape: [usize; 4], layout: ConvLayout) -> [usize; 4] {
    match layout {
        ConvLayout::Nchw => shape,
        ConvLayout::Nhwc => [shape[0], shape[3], shape[1], shape[2]],
    }
}

/// Shape in `layout` order of canonical extents.
pub fn from_canonical(dims: [usize; 4], layout: ConvLayout) -> [usize; 4] {
    match layout {
        ConvLayout::Nchw => dims,
        ConvLayout::Nhwc => [dims[0], dims[2], dims[3], dims[1]],
    }
}

/// Element index of canonical coordinates in a contiguous tensor of `layout`.
fn linear(dims: [usize; 4], [n, c, h, w]: [usize; 4], layout: ConvLayout) -> usize {
    let [_, cd, hd, wd] = dims;
    match layout {
        ConvLayout::Nchw => ((n * cd + c) * hd + h) * wd + w,
        ConvLayout::Nhwc => ((n * hd + h) * wd + w) * cd + c,
    }
}

/// Output canonical extents of `op` on canonical input extents.
pub fn out_dims(op: &RefOp, [n, c, h, w]: [usize; 4]) -> [usize; 4] {
    match op {
        RefOp::Shuffle(r) => [n, c / (r * r), h * r, w * r],
        RefOp::Unshuffle(r) => [n, c * r * r, h / r, w / r],
        RefOp::Nearest { rows, cols } => [n, c, rows.len(), cols.len()],
    }
}

/// Canonical source coordinates of canonical output coordinates.
fn source(op: &RefOp, [n, c, h, w]: [usize; 4]) -> [usize; 4] {
    match op {
        RefOp::Shuffle(r) => [n, c * r * r + (h % r) * r + w % r, h / r, w / r],
        RefOp::Unshuffle(r) => {
            let (i, j) = ((c % (r * r)) / r, c % r);
            [n, c / (r * r), h * r + i, w * r + j]
        }
        RefOp::Nearest { rows, cols } => [n, c, rows[h], cols[w]],
    }
}

/// Apply `op` to contiguous `input` bytes of `shape` (in `layout` order).
///
/// Returns the output bytes and the output shape in `layout` order.
pub fn reference(
    input: &[u8],
    elem: usize,
    shape: [usize; 4],
    layout: ConvLayout,
    op: &RefOp,
) -> (Vec<u8>, [usize; 4]) {
    let in_dims = to_canonical(shape, layout);
    let dims = out_dims(op, in_dims);
    let numel: usize = dims.iter().product();
    let mut out = vec![0u8; numel * elem];
    for n in 0..dims[0] {
        for c in 0..dims[1] {
            for h in 0..dims[2] {
                for w in 0..dims[3] {
                    let d = linear(dims, [n, c, h, w], layout) * elem;
                    let s = linear(in_dims, source(op, [n, c, h, w]), layout) * elem;
                    out[d..d + elem].copy_from_slice(&input[s..s + elem]);
                }
            }
        }
    }
    (out, from_canonical(dims, layout))
}

/// PyTorch CPU `nearest_idx` table for one axis, computed here from the
/// formula independently of numr.
pub fn nearest_table(in_size: usize, out_size: usize, scale_factor: Option<f64>) -> Vec<usize> {
    (0..out_size)
        .map(|dst| {
            if out_size == in_size {
                dst
            } else if out_size == 2 * in_size {
                dst >> 1
            } else {
                let scale: f32 = match scale_factor {
                    Some(s) => (1.0 / s) as f32,
                    None => in_size as f32 / out_size as f32,
                };
                (((dst as f32) * scale).floor() as usize).min(in_size - 1)
            }
        })
        .collect()
}

/// Output extent `floor(in * s)` of a scale factor, in f64.
pub fn scaled_extent(in_size: usize, s: f64) -> usize {
    (in_size as f64 * s).floor() as usize
}

/// Deterministic element bytes. Element `i`, byte `k` is
/// `(i * 31 + k * 7 + 1) % 251`. Bool elements hold 0 or 1.
pub fn pattern(numel: usize, dtype: DType) -> Vec<u8> {
    let elem = dtype.size_in_bytes();
    (0..numel * elem)
        .map(|b| {
            let (i, k) = (b / elem, b % elem);
            if dtype == DType::Bool {
                (i % 2) as u8
            } else {
                ((i * 31 + k * 7 + 1) % 251) as u8
            }
        })
        .collect()
}

/// Assert equal bytes and name the first differing element.
pub fn assert_bytes_eq(got: &[u8], want: &[u8], elem: usize, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: byte length differs");
    if let Some(b) = got.iter().zip(want).position(|(g, w)| g != w) {
        let i = b / elem;
        panic!(
            "{label}: element {i} differs: got {:?}, want {:?}",
            &got[i * elem..(i + 1) * elem],
            &want[i * elem..(i + 1) * elem]
        );
    }
}

/// Every dtype numr defines.
pub const ALL_DTYPES: [DType; 17] = [
    DType::F64,
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::FP8E4M3,
    DType::FP8E5M2,
    DType::I64,
    DType::I32,
    DType::I16,
    DType::I8,
    DType::U64,
    DType::U32,
    DType::U16,
    DType::U8,
    DType::Bool,
    DType::Complex64,
    DType::Complex128,
];
