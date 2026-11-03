//! Per-call checks of the packed RGB24 `ColorOps` methods, shared by every
//! backend.

use super::color_calls::{check_float, plan_pack};
use super::color_packed::{RGB24_COL_BYTES, packed_rgb24_frame_size};
use super::color_validation::{check_disjoint, check_writable, rgb_shape, validate_out};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::traits::{ConvLayout, PackedPlane};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Shape `[H, ceil(3·W / 4)]` of the U32 words a new packed RGB24 frame of
/// `[H, W]` takes.
pub fn packed_rgb24_shape([h, w]: [usize; 2]) -> [usize; 2] {
    [h, (RGB24_COL_BYTES * w).div_ceil(4)]
}

/// Check a `packed_rgb24_to_float` call and return `[H, W]`.
///
/// # Errors
///
/// The `rgb24` errors of `packed_rgb24_frame_size`, and
/// `Error::UnsupportedDType` for `out_dtype`.
pub fn plan_unpack_packed<R: Runtime<DType = DType>, F: Fn(DType) -> bool>(
    rgb24: &PackedPlane<'_, R>,
    out_dtype: DType,
    float_ok: F,
    op: &'static str,
) -> Result<[usize; 2]> {
    let size = packed_rgb24_frame_size(rgb24, "rgb24", op)?;
    check_float(out_dtype, float_ok, op)?;
    Ok(size)
}

/// Check a `packed_rgb24_to_float_into` call and return `[H, W]`.
///
/// # Errors
///
/// Every error of [`plan_unpack_packed`], plus the `out` errors of
/// `validate_out`.
pub fn plan_unpack_packed_into<R: Runtime<DType = DType>, F: Fn(DType) -> bool>(
    out: &Tensor<R>,
    rgb24: &PackedPlane<'_, R>,
    layout: ConvLayout,
    float_ok: F,
    op: &'static str,
) -> Result<[usize; 2]> {
    let [h, w] = plan_unpack_packed(rgb24, out.dtype(), float_ok, op)?;
    validate_out(out, &rgb_shape(layout, h, w), &[(rgb24.words, "rgb24")], op)?;
    Ok([h, w])
}

/// Check a `float_to_packed_rgb24_into` call and return `[H, W]`.
///
/// # Errors
///
/// The `rgb` errors of `plan_pack`, the `out` errors of
/// `packed_rgb24_frame_size`, `Error::ShapeMismatch` when `out` holds
/// another `[H, W]`, and `Error::InvalidArgument` for an `out` that repeats
/// a word or overlaps `rgb`.
pub fn plan_pack_packed_into<R: Runtime<DType = DType>, F: Fn(DType) -> bool>(
    out: &PackedPlane<'_, R>,
    rgb: &Tensor<R>,
    layout: ConvLayout,
    float_ok: F,
    op: &'static str,
) -> Result<[usize; 2]> {
    let size = plan_pack(rgb, layout, float_ok, op)?;
    let got = packed_rgb24_frame_size(out, "out", op)?;
    if got != size {
        return Err(Error::shape_mismatch(&size, &got));
    }
    check_writable(out.words, "out", "out", op)?;
    check_disjoint((out.words, "out"), (rgb, "rgb"), "out", op)?;
    Ok(size)
}
