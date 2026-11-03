//! Coefficients and plane geometry of the `ColorOps` conversions.
//!
//! [`ColorCoeffs`] holds every constant of the documented formulas, each
//! computed in f64 and rounded to f32 once. The CPU kernels and the CUDA
//! kernels read the same struct, so both evaluate the same f32 operations.
//! [`YuvFrame`] holds the planes as addresses and strides, with an NV12 or
//! P010 chroma plane split into a U view and a V view, so one kernel serves
//! every format.

use super::color_validation::validate_planes;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::traits::{ChromaSiting, ColorRange, ConvLayout, YuvPlanes, YuvSpec};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Constants of one matrix, range and bit depth, in f32.
///
/// The layout is `#[repr(C)]` and matches `ColorCoeffs` in `color.cu`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorCoeffs {
    /// Luma code of black: `16·s`, or 0 for full range.
    pub y_off: i32,
    /// Chroma code of zero: `128·s`, or `2^(bits-1)` for full range.
    pub c_off: i32,
    /// `16·224·s`, or `16·(2^bits - 1)`: the chroma divisor for sums in
    /// sixteenths.
    pub c_div16: f32,
    /// `2(1 - Kr)`.
    pub cr_r: f32,
    /// `2(1 - Kb)`.
    pub cb_b: f32,
    /// `-2·Kb·(1 - Kb) / Kg`.
    pub g_cb: f32,
    /// `-2·Kr·(1 - Kr) / Kg`.
    pub g_cr: f32,
    /// `Kr`.
    pub kr: f32,
    /// `Kg = 1 - Kr - Kb`.
    pub kg: f32,
    /// `Kb`.
    pub kb: f32,
    /// `219·s`, or `2^bits - 1`: the luma divisor and multiplier.
    pub y_mul: f32,
    /// `224·s`, or `2^bits - 1`.
    pub c_mul: f32,
    /// Lowest luma code.
    pub y_lo: f32,
    /// Highest luma code.
    pub y_hi: f32,
    /// Lowest chroma code.
    pub c_lo: f32,
    /// Highest chroma code.
    pub c_hi: f32,
}

impl ColorCoeffs {
    /// Constants of `spec` at `bits` per sample. `bits` is 8 or 10.
    pub fn new(spec: &YuvSpec, bits: u32) -> Self {
        let (kr, kb) = spec.matrix.kr_kb();
        let kg = 1.0 - kr - kb;
        let s = f64::from(1u32 << bits.saturating_sub(8));
        let max = f64::from((1u32 << bits) - 1);
        let (y_off, c_off, y_mul, c_mul, y_lo, y_hi, c_lo, c_hi) = match spec.range {
            ColorRange::Limited => (
                16.0 * s,
                128.0 * s,
                219.0 * s,
                224.0 * s,
                16.0 * s,
                235.0 * s,
                16.0 * s,
                240.0 * s,
            ),
            ColorRange::Full => (0.0, (max + 1.0) / 2.0, max, max, 0.0, max, 0.0, max),
        };
        Self {
            y_off: y_off as i32,
            c_off: c_off as i32,
            c_div16: (16.0 * c_mul) as f32,
            cr_r: (2.0 * (1.0 - kr)) as f32,
            cb_b: (2.0 * (1.0 - kb)) as f32,
            g_cb: (-2.0 * kb * (1.0 - kb) / kg) as f32,
            g_cr: (-2.0 * kr * (1.0 - kr) / kg) as f32,
            kr: kr as f32,
            kg: kg as f32,
            kb: kb as f32,
            y_mul: y_mul as f32,
            c_mul: c_mul as f32,
            y_lo: y_lo as f32,
            y_hi: y_hi as f32,
            c_lo: c_lo as f32,
            c_hi: c_hi as f32,
        }
    }
}

/// One plane as the address of element `[0, 0]` and strides in elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneView {
    /// Address of element `[0, 0]`.
    pub addr: u64,
    /// Stride between rows.
    pub row_stride: isize,
    /// Stride between columns.
    pub col_stride: isize,
}

/// Validated geometry of one 4:2:0 frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YuvFrame {
    /// Luma rows `H`.
    pub height: usize,
    /// Luma columns `W`.
    pub width: usize,
    /// Samples are U16 (P010) rather than U8.
    pub wide: bool,
    /// Right shift from a stored sample to its value: 6 for P010, else 0.
    pub shift: u32,
    /// Horizontal chroma siting is `Center`.
    pub center: bool,
    /// Luma plane.
    pub y: PlaneView,
    /// Cb samples, `[H/2, W/2]`.
    pub u: PlaneView,
    /// Cr samples, `[H/2, W/2]`.
    pub v: PlaneView,
    /// Constants of the spec at the frame's bit depth.
    pub coeffs: ColorCoeffs,
}

impl YuvFrame {
    /// Pixel count `H·W`.
    pub fn pixels(&self) -> usize {
        self.height * self.width
    }
}

fn plane_view<R: Runtime<DType = DType>>(t: &Tensor<R>) -> PlaneView {
    PlaneView {
        addr: t.ptr(),
        row_stride: t.strides()[0],
        col_stride: t.strides()[1],
    }
}

/// U and V views of an interleaved chroma plane: U at even columns, V at odd.
fn split_interleaved<R: Runtime<DType = DType>>(
    uv: &Tensor<R>,
    op: &'static str,
) -> Result<(PlaneView, PlaneView)> {
    let base = plane_view(uv);
    let elem = uv.dtype().size_in_bytes() as i64;
    let pair = base.col_stride.checked_mul(2);
    let v_shift = (base.col_stride as i64).checked_mul(elem);
    let (Some(pair), Some(v_shift)) = (pair, v_shift) else {
        return Err(Error::InvalidArgument {
            arg: "planes",
            reason: format!(
                "{op}: plane uv column stride {} overflows when doubled; pass a view with a \
                 smaller column stride",
                base.col_stride
            ),
        });
    };
    let u = PlaneView {
        col_stride: pair,
        ..base
    };
    let v = PlaneView {
        addr: base.addr.wrapping_add_signed(v_shift),
        ..u
    };
    Ok((u, v))
}

/// Check `planes` and build the frame geometry for `spec`.
///
/// # Errors
///
/// Every error of [`validate_planes`], and `Error::InvalidArgument` for a
/// chroma column stride that overflows when doubled.
pub fn plan_yuv_frame<R: Runtime<DType = DType>>(
    planes: &YuvPlanes<'_, R>,
    spec: &YuvSpec,
    op: &'static str,
) -> Result<YuvFrame> {
    let [height, width] = validate_planes(planes, op)?;
    let (u, v) = match *planes {
        YuvPlanes::Nv12 { uv, .. } | YuvPlanes::P010 { uv, .. } => split_interleaved(uv, op)?,
        YuvPlanes::I420 { u, v, .. } => (plane_view(u), plane_view(v)),
    };
    let wide = matches!(planes, YuvPlanes::P010 { .. });
    Ok(YuvFrame {
        height,
        width,
        wide,
        shift: if wide { 6 } else { 0 },
        center: spec.siting == ChromaSiting::Center,
        y: plane_view(planes.y()),
        u,
        v,
        coeffs: ColorCoeffs::new(spec, planes.bits()),
    })
}

/// Strides of a float RGB frame in `layout`, as `[C, H, W]`.
pub fn rgb_strides(strides: &[isize], layout: ConvLayout) -> [isize; 3] {
    match layout {
        ConvLayout::Nchw => [strides[1], strides[2], strides[3]],
        ConvLayout::Nhwc => [strides[3], strides[1], strides[2]],
    }
}

/// Strides of a U8 `[H, W, 3]` RGB24 frame, as `[C, H, W]`.
pub fn rgb24_strides(strides: &[isize]) -> [isize; 3] {
    [strides[2], strides[0], strides[1]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::traits::ColorMatrix;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn limited_8bit_constants() {
        let spec = YuvSpec::new(ColorMatrix::Bt709, ColorRange::Limited, ChromaSiting::Left);
        let k = ColorCoeffs::new(&spec, 8);
        assert_eq!((k.y_off, k.c_off), (16, 128));
        assert_eq!((k.y_mul, k.c_mul), (219.0, 224.0));
        assert_eq!((k.y_lo, k.y_hi, k.c_lo, k.c_hi), (16.0, 235.0, 16.0, 240.0));
        assert_eq!(k.cr_r, 1.5748);
        assert_eq!(k.cb_b, 1.8556);
    }

    #[test]
    fn full_10bit_constants() {
        let spec = YuvSpec::new(ColorMatrix::Bt601, ColorRange::Full, ChromaSiting::Center);
        let k = ColorCoeffs::new(&spec, 10);
        assert_eq!((k.y_off, k.c_off), (0, 512));
        assert_eq!((k.y_mul, k.y_hi, k.c_hi), (1023.0, 1023.0, 1023.0));
        assert_eq!(k.c_div16, 16.0 * 1023.0);
    }

    #[test]
    fn interleaved_chroma_splits_into_u_and_v() {
        let uv = Tensor::<CpuRuntime>::zeros(&[2, 8], DType::U16, &CpuDevice::new()).unwrap();
        let (u, v) = split_interleaved(&uv, "t").unwrap();
        assert_eq!((u.row_stride, u.col_stride), (8, 2));
        assert_eq!(v.addr, u.addr + 2);
        assert_eq!(v.col_stride, 2);
    }
}
