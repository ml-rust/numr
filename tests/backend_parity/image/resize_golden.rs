// PyTorch CPU references for ImageOps::resize, read from resize_golden.bin.
//
// gen_resize_golden.py writes the file and documents its layout. The input is
// rebuilt here from the same index formula, so only the outputs are stored.

use numr::ops::{ConvLayout, ResizeConfig, ResizeFilter};

const GOLDEN: &[u8] = include_bytes!("resize_golden.bin");

/// Input dtype a block's reference was taken from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// F32 input.
    F32,
    /// Input rounded to F16, reference in F32.
    F16,
    /// Input rounded to BF16, reference in F32.
    BF16,
}

/// One reference: input extents, config and the NCHW output.
#[derive(Debug, Clone)]
pub struct Block {
    pub kind: Kind,
    /// Input extents as `[N, C, H, W]`.
    pub dims: [usize; 4],
    pub size: (usize, usize),
    pub filter: ResizeFilter,
    pub antialias: bool,
    /// NCHW output.
    pub want: Vec<f32>,
}

impl Block {
    pub fn config(&self) -> ResizeConfig {
        ResizeConfig::new(self.size, self.filter).with_antialias(self.antialias)
    }

    /// Output extents as `[N, C, H, W]`.
    pub fn out_dims(&self) -> [usize; 4] {
        [self.dims[0], self.dims[1], self.size.0, self.size.1]
    }

    pub fn label(&self) -> String {
        format!(
            "{:?} {:?} -> {:?} {:?} antialias={}",
            self.kind, self.dims, self.size, self.filter, self.antialias
        )
    }
}

fn read_u32(bytes: &[u8], at: &mut usize) -> u32 {
    let v = u32::from_le_bytes(bytes[*at..*at + 4].try_into().expect("4 bytes"));
    *at += 4;
    v
}

/// Every block of the fixture.
pub fn blocks() -> Vec<Block> {
    assert_eq!(&GOLDEN[..4], b"NRSZ", "resize_golden.bin: bad magic");
    let mut at = 4;
    assert_eq!(read_u32(GOLDEN, &mut at), 1, "resize_golden.bin: version");
    let count = read_u32(GOLDEN, &mut at) as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let h: Vec<usize> = (0..9).map(|_| read_u32(GOLDEN, &mut at) as usize).collect();
        let kind = match h[0] {
            0 => Kind::F32,
            1 => Kind::F16,
            2 => Kind::BF16,
            k => panic!("resize_golden.bin: unknown kind {k}"),
        };
        let filter = if h[7] == 0 {
            ResizeFilter::Bilinear
        } else {
            ResizeFilter::Bicubic
        };
        let numel = h[1] * h[2] * h[5] * h[6];
        let want = GOLDEN[at..at + 4 * numel]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        at += 4 * numel;
        out.push(Block {
            kind,
            dims: [h[1], h[2], h[3], h[4]],
            size: (h[5], h[6]),
            filter,
            antialias: h[8] != 0,
            want,
        });
    }
    assert_eq!(at, GOLDEN.len(), "resize_golden.bin: trailing bytes");
    out
}

/// The generator's input: `f32(k) / 4099 - 0.25` with
/// `k = (i * 7919 + 104729) % 4099` over the NCHW index `i`.
pub fn input(dims: [usize; 4]) -> Vec<f32> {
    let n: usize = dims.iter().product();
    (0..n as u64)
        .map(|i| ((i * 7919 + 104729) % 4099) as f32 / 4099.0f32 - 0.25f32)
        .collect()
}

/// `data` in NCHW order rearranged into `layout` order.
pub fn to_layout(data: &[f32], [n, c, h, w]: [usize; 4], layout: ConvLayout) -> Vec<f32> {
    match layout {
        ConvLayout::Nchw => data.to_vec(),
        ConvLayout::Nhwc => {
            let mut out = vec![0.0; data.len()];
            for ni in 0..n {
                for ci in 0..c {
                    for hi in 0..h {
                        for wi in 0..w {
                            let src = ((ni * c + ci) * h + hi) * w + wi;
                            out[((ni * h + hi) * w + wi) * c + ci] = data[src];
                        }
                    }
                }
            }
            out
        }
    }
}

/// `data` in `layout` order rearranged into NCHW order.
pub fn from_layout(data: &[f32], [n, c, h, w]: [usize; 4], layout: ConvLayout) -> Vec<f32> {
    match layout {
        ConvLayout::Nchw => data.to_vec(),
        ConvLayout::Nhwc => {
            let mut out = vec![0.0; data.len()];
            for ni in 0..n {
                for ci in 0..c {
                    for hi in 0..h {
                        for wi in 0..w {
                            let dst = ((ni * c + ci) * h + hi) * w + wi;
                            out[dst] = data[((ni * h + hi) * w + wi) * c + ci];
                        }
                    }
                }
            }
            out
        }
    }
}

/// Shape of canonical extents `dims` in `layout` order.
pub fn shape(dims: [usize; 4], layout: ConvLayout) -> [usize; 4] {
    match layout {
        ConvLayout::Nchw => dims,
        ConvLayout::Nhwc => [dims[0], dims[2], dims[3], dims[1]],
    }
}

/// Unit roundoff of f32.
const U: f32 = f32::EPSILON / 2.0;

/// Taps per output of one pass, or 0 when antialias skips the axis.
fn taps(in_size: usize, out_size: usize, filter: ResizeFilter, antialias: bool) -> f32 {
    let half = if filter == ResizeFilter::Bicubic {
        2.0
    } else {
        1.0
    };
    if antialias {
        if in_size == out_size {
            return 0.0;
        }
        let support = half * (in_size as f64 / out_size as f64).max(1.0);
        (2.0 * support.ceil() + 1.0) as f32
    } else {
        2.0 * half as f32
    }
}

/// Absolute bound on `|got - want|` for an F32 result against the torch
/// reference `want`.
///
/// - Accumulation: a pass sums `n` products. Two summation orders, or a
///   fused and an unfused step, differ by at most `2·n·u·Σ|x·w|`, with
///   `Σ|x·w| <= M·L`, `M = max|x|` and `L = 1.5 >= Σ|w|` for every filter.
///   The H pass also carries the W pass difference times `L`. A factor 2 on
///   top covers weight rounding. That gives `4·u·M·L·(n_w·L + n_h)`.
/// - Source index, without antialias: PyTorch's small-output bilinear
///   kernel fuses `scale·(i + 0.5) − 0.5`, so `λ` can move by `u·in`, which
///   moves the output by `λ` times the input range. That gives
///   `2·u·(in_h + in_w)·range` for every plain filter.
/// - Relative: `4·u·|want|` for the final rounding.
pub fn tolerance(block: &Block, x: &[f32], want: f32) -> f32 {
    let m = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let lo = x.iter().fold(f32::INFINITY, |a, &v| a.min(v));
    let hi = x.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
    let l = 1.5f32;
    let [_, _, h, w] = block.dims;
    let (oh, ow) = block.size;
    let n_w = taps(w, ow, block.filter, block.antialias);
    let n_h = taps(h, oh, block.filter, block.antialias);
    let acc = 4.0 * U * m * l * (n_w * l + n_h);
    let index = if !block.antialias {
        2.0 * U * (h + w) as f32 * (hi - lo)
    } else {
        0.0
    };
    acc + index + 4.0 * U * want.abs()
}

/// One unit in the last place of `v` in a storage format with `mantissa`
/// explicit bits (10 for F16, 7 for BF16).
#[cfg(feature = "f16")]
pub fn storage_ulp(v: f32, mantissa: i32) -> f32 {
    let a = v.abs().max(f32::MIN_POSITIVE);
    2f32.powi(a.log2().floor() as i32 - mantissa)
}

/// Assert `got` matches `want` within `bound(i)` at every element.
pub fn assert_close(got: &[f32], want: &[f32], bound: impl Fn(usize) -> f32, label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (&g, &e)) in got.iter().zip(want).enumerate() {
        let tol = bound(i);
        assert!(
            (g - e).abs() <= tol,
            "{label}: element {i}: got {g:e}, want {e:e}, |diff| {:e} > {tol:e}",
            (g - e).abs()
        );
    }
}
