// numpy references for ColorOps, read from color_golden.bin.
//
// gen_color_golden.py writes the file and documents its layout.

use numr::dtype::DType;
use numr::ops::{ChromaSiting, ColorMatrix, ColorRange, YuvSpec};

const GOLDEN: &[u8] = include_bytes!("color_golden.bin");

/// Plane format of a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Nv12,
    I420,
    P010,
}

impl Format {
    pub const ALL: [Format; 3] = [Format::Nv12, Format::I420, Format::P010];

    /// Sample dtype of every plane.
    pub fn dtype(self) -> DType {
        match self {
            Format::P010 => DType::U16,
            _ => DType::U8,
        }
    }

    /// Bits per sample value.
    pub fn bits(self) -> u32 {
        match self {
            Format::P010 => 10,
            _ => 8,
        }
    }

    /// Plane shapes of an `h x w` frame, luma first.
    pub fn plane_shapes(self, h: usize, w: usize) -> Vec<[usize; 2]> {
        match self {
            Format::I420 => vec![[h, w], [h / 2, w / 2], [h / 2, w / 2]],
            _ => vec![[h, w], [h / 2, w]],
        }
    }
}

/// Format, spec and size of one block.
#[derive(Debug, Clone, Copy)]
pub struct Case {
    pub format: Format,
    pub spec: YuvSpec,
    pub h: usize,
    pub w: usize,
}

/// One `yuv_to_rgb` reference: stored plane samples and RGB in `[3, H, W]`.
#[derive(Debug, Clone)]
pub struct DecodeBlock {
    pub case: Case,
    pub planes: Vec<Vec<u16>>,
    pub rgb: Vec<f32>,
}

/// One `rgb_to_yuv_into` reference: RGB in `[3, H, W]` and stored samples.
#[derive(Debug, Clone)]
pub struct EncodeBlock {
    pub case: Case,
    pub rgb: Vec<f32>,
    pub planes: Vec<Vec<u16>>,
}

/// A BT.709 limited NV12 frame and ffmpeg's rgb24 conversion of it.
#[derive(Debug, Clone)]
pub struct FfmpegBlock {
    pub h: usize,
    pub w: usize,
    pub planes: Vec<Vec<u16>>,
    pub rgb24: Vec<u8>,
}

/// Every reference of the fixture.
#[derive(Debug, Clone)]
pub struct Golden {
    pub decode: Vec<DecodeBlock>,
    pub encode: Vec<EncodeBlock>,
    /// `(x, want)`: `x · 255` is exactly `k + 0.5` in f32, `want` is even.
    pub ties: Vec<(f32, u8)>,
    pub ffmpeg: Option<FfmpegBlock>,
}

struct Reader {
    at: usize,
}

impl Reader {
    fn take(&mut self, n: usize) -> &'static [u8] {
        let s = &GOLDEN[self.at..self.at + n];
        self.at += n;
        s
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().expect("4 bytes"))
    }

    fn f32s(&mut self, n: usize) -> Vec<f32> {
        let bytes = self.take(4 * n);
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }

    fn samples(&mut self, n: usize, dtype: DType) -> Vec<u16> {
        match dtype {
            DType::U16 => self
                .take(2 * n)
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes(*b))
                .collect(),
            _ => self.take(n).iter().map(|&b| u16::from(b)).collect(),
        }
    }

    fn planes(&mut self, format: Format, h: usize, w: usize) -> Vec<Vec<u16>> {
        format
            .plane_shapes(h, w)
            .iter()
            .map(|[r, c]| self.samples(r * c, format.dtype()))
            .collect()
    }

    fn case(&mut self) -> Case {
        let h: Vec<u32> = (0..6).map(|_| self.u32()).collect();
        let format = Format::ALL[h[0] as usize];
        let matrix = [ColorMatrix::Bt601, ColorMatrix::Bt709][h[1] as usize];
        let range = [ColorRange::Limited, ColorRange::Full][h[2] as usize];
        let siting = [ChromaSiting::Left, ChromaSiting::Center][h[3] as usize];
        Case {
            format,
            spec: YuvSpec::new(matrix, range, siting),
            h: h[4] as usize,
            w: h[5] as usize,
        }
    }
}

/// Parse the fixture.
pub fn golden() -> Golden {
    assert_eq!(&GOLDEN[..4], b"NCLR", "color_golden.bin: bad magic");
    let mut r = Reader { at: 4 };
    assert_eq!(r.u32(), 1, "color_golden.bin: version");
    let decode = (0..r.u32())
        .map(|_| {
            let case = r.case();
            let planes = r.planes(case.format, case.h, case.w);
            let rgb = r.f32s(3 * case.h * case.w);
            DecodeBlock { case, planes, rgb }
        })
        .collect();
    let encode = (0..r.u32())
        .map(|_| {
            let case = r.case();
            let rgb = r.f32s(3 * case.h * case.w);
            let planes = r.planes(case.format, case.h, case.w);
            EncodeBlock { case, rgb, planes }
        })
        .collect();
    let ties = (0..r.u32())
        .map(|_| {
            let x = r.f32s(1)[0];
            (x, r.u32() as u8)
        })
        .collect();
    let ffmpeg = (r.u32() == 1).then(|| {
        let (h, w) = (r.u32() as usize, r.u32() as usize);
        let planes = r.planes(Format::Nv12, h, w);
        let rgb24 = r.take(h * w * 3).to_vec();
        FfmpegBlock {
            h,
            w,
            planes,
            rgb24,
        }
    });
    assert_eq!(r.at, GOLDEN.len(), "color_golden.bin: trailing bytes");
    Golden {
        decode,
        encode,
        ties,
        ffmpeg,
    }
}
