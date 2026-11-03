//! Colour spec and plane types of [`ColorOps`](super::ColorOps).

use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// YCbCr matrix: the luma weights `Kr` and `Kb`, with `Kg = 1 - Kr - Kb`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColorMatrix {
    /// ITU-R BT.601: `Kr = 0.299`, `Kb = 0.114`.
    Bt601,
    /// ITU-R BT.709: `Kr = 0.2126`, `Kb = 0.0722`.
    Bt709,
}

impl ColorMatrix {
    /// `(Kr, Kb)` of the matrix.
    pub fn kr_kb(self) -> (f64, f64) {
        match self {
            Self::Bt601 => (0.299, 0.114),
            Self::Bt709 => (0.2126, 0.0722),
        }
    }
}

/// Code range of the Y, Cb and Cr samples.
///
/// With `bits` the sample depth and `s = 2^(bits - 8)`:
///
/// | Range | Y codes | Cb, Cr codes |
/// | --- | --- | --- |
/// | `Limited` | `16·s ..= 235·s` | `16·s ..= 240·s` |
/// | `Full` | `0 ..= 2^bits - 1` | `0 ..= 2^bits - 1` |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColorRange {
    /// Studio range, also called TV or MPEG range.
    Limited,
    /// Full range, also called PC or JPEG range.
    Full,
}

/// Horizontal position of a 4:2:0 chroma sample.
///
/// Vertically, chroma row `i` sits midway between luma rows `2i` and
/// `2i + 1` for both sitings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ChromaSiting {
    /// Chroma column `j` sits on luma column `2j`. This is MPEG-2 "left",
    /// the H.264, HEVC and AV1 default.
    Left,
    /// Chroma column `j` sits midway between luma columns `2j` and `2j + 1`.
    /// This is MPEG-1 and JPEG siting.
    Center,
}

/// Matrix, range and chroma siting of a YCbCr frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct YuvSpec {
    /// Luma weights.
    pub matrix: ColorMatrix,
    /// Code range.
    pub range: ColorRange,
    /// Horizontal chroma siting.
    pub siting: ChromaSiting,
}

impl YuvSpec {
    /// Spec with `matrix`, `range` and `siting`.
    pub fn new(matrix: ColorMatrix, range: ColorRange, siting: ChromaSiting) -> Self {
        Self {
            matrix,
            range,
            siting,
        }
    }
}

/// The planes of one 4:2:0 frame of `H x W` luma samples.
///
/// `H` and `W` must be even. Each plane is a 2-D view `[rows, cols]` with
/// any strides, so a surface with a row pitch passes as a strided view.
///
/// | Format | Y | Chroma | Bits |
/// | --- | --- | --- | --- |
/// | `Nv12` | U8 `[H, W]` | `uv`: U8 `[H/2, W]`, interleaved `U, V` | 8 |
/// | `I420` | U8 `[H, W]` | `u`, `v`: U8 `[H/2, W/2]` each | 8 |
/// | `P010` | U16 `[H, W]` | `uv`: U16 `[H/2, W]`, interleaved `U, V` | 10 |
///
/// A P010 sample holds its 10-bit value in the high bits: `value = raw >> 6`.
/// A read ignores the low 6 bits, and a write stores `value << 6`.
#[non_exhaustive]
pub enum YuvPlanes<'a, R: Runtime> {
    /// 8-bit luma plane and one interleaved chroma plane.
    Nv12 {
        /// Luma, U8 `[H, W]`.
        y: &'a Tensor<R>,
        /// Chroma, U8 `[H/2, W]`. Column `2j` holds U and `2j + 1` holds V.
        uv: &'a Tensor<R>,
    },
    /// 8-bit luma plane and two chroma planes.
    I420 {
        /// Luma, U8 `[H, W]`.
        y: &'a Tensor<R>,
        /// Cb, U8 `[H/2, W/2]`.
        u: &'a Tensor<R>,
        /// Cr, U8 `[H/2, W/2]`.
        v: &'a Tensor<R>,
    },
    /// 10-bit luma plane and one interleaved chroma plane, in U16 samples.
    P010 {
        /// Luma, U16 `[H, W]`.
        y: &'a Tensor<R>,
        /// Chroma, U16 `[H/2, W]`. Column `2j` holds U and `2j + 1` holds V.
        uv: &'a Tensor<R>,
    },
}

impl<R: Runtime> Clone for YuvPlanes<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime> Copy for YuvPlanes<'_, R> {}

impl<R: Runtime> std::fmt::Debug for YuvPlanes<'_, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.format_name())
    }
}

impl<'a, R: Runtime> YuvPlanes<'a, R> {
    /// Format name for errors: `"NV12"`, `"I420"` or `"P010"`.
    pub fn format_name(&self) -> &'static str {
        match self {
            Self::Nv12 { .. } => "NV12",
            Self::I420 { .. } => "I420",
            Self::P010 { .. } => "P010",
        }
    }

    /// Sample depth in bits: 8 for NV12 and I420, 10 for P010.
    pub fn bits(&self) -> u32 {
        match self {
            Self::Nv12 { .. } | Self::I420 { .. } => 8,
            Self::P010 { .. } => 10,
        }
    }

    /// Luma plane.
    pub fn y(&self) -> &'a Tensor<R> {
        match *self {
            Self::Nv12 { y, .. } | Self::I420 { y, .. } | Self::P010 { y, .. } => y,
        }
    }
}
