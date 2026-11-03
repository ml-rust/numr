//! Per-call checks of the `ColorOps` methods, shared by every backend.
//!
//! Each function checks every argument of one method and returns what the
//! kernels need. A backend passes its own float dtype test as `float_ok`.

use super::color_plan::{YuvFrame, plan_yuv_frame};
use super::color_validation::{
    named_planes, rgb_frame_size, rgb_shape, rgb24_frame_size, validate_out,
    validate_written_planes,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::traits::{ConvLayout, YuvPlanes, YuvSpec};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Return `Error::UnsupportedDType` unless `float_ok(dtype)`.
pub fn check_float<F: Fn(DType) -> bool>(
    dtype: DType,
    float_ok: F,
    op: &'static str,
) -> Result<()> {
    if float_ok(dtype) {
        Ok(())
    } else {
        Err(Error::UnsupportedDType { dtype, op })
    }
}

/// Check a `yuv_to_rgb` call and return the frame.
///
/// # Errors
///
/// The `planes` errors of `plan_yuv_frame`, and `Error::UnsupportedDType`
/// for `out_dtype`.
pub fn plan_decode<R: Runtime<DType = DType>, F: Fn(DType) -> bool>(
    planes: &YuvPlanes<'_, R>,
    spec: &YuvSpec,
    out_dtype: DType,
    float_ok: F,
    op: &'static str,
) -> Result<YuvFrame> {
    let frame = plan_yuv_frame(planes, spec, op)?;
    check_float(out_dtype, float_ok, op)?;
    Ok(frame)
}

/// Check a `yuv_to_rgb_into` call and return the frame.
///
/// # Errors
///
/// Every error of [`plan_decode`], plus the `out` errors of `validate_out`.
pub fn plan_decode_into<R: Runtime<DType = DType>, F: Fn(DType) -> bool>(
    out: &Tensor<R>,
    planes: &YuvPlanes<'_, R>,
    spec: &YuvSpec,
    layout: ConvLayout,
    float_ok: F,
    op: &'static str,
) -> Result<YuvFrame> {
    let frame = plan_decode(planes, spec, out.dtype(), float_ok, op)?;
    let want = rgb_shape(layout, frame.height, frame.width);
    validate_out(out, &want, &named_planes(planes), op)?;
    Ok(frame)
}

/// Check an `rgb_to_yuv_into` call and return the frame.
///
/// # Errors
///
/// The `planes` errors of `plan_yuv_frame` and `validate_written_planes`,
/// `Error::InvalidArgument` for an `rgb` that is not one frame,
/// `Error::ShapeMismatch` for an `rgb` of another size, and
/// `Error::UnsupportedDType` for its dtype.
pub fn plan_encode<R: Runtime<DType = DType>, F: Fn(DType) -> bool>(
    rgb: &Tensor<R>,
    planes: &YuvPlanes<'_, R>,
    spec: &YuvSpec,
    layout: ConvLayout,
    float_ok: F,
    op: &'static str,
) -> Result<YuvFrame> {
    let frame = plan_yuv_frame(planes, spec, op)?;
    let size = rgb_frame_size(rgb, layout, "rgb", op)?;
    if size != [frame.height, frame.width] {
        let want = rgb_shape(layout, frame.height, frame.width);
        return Err(Error::shape_mismatch(&want, rgb.shape()));
    }
    check_float(rgb.dtype(), float_ok, op)?;
    validate_written_planes(planes, rgb, op)?;
    Ok(frame)
}

/// Check an `rgb24_to_float` call and return `[H, W]`.
///
/// # Errors
///
/// `Error::InvalidArgument` for an `rgb24` that is not U8 `[H, W, 3]`, and
/// `Error::UnsupportedDType` for `out_dtype`.
pub fn plan_unpack<R: Runtime<DType = DType>, F: Fn(DType) -> bool>(
    rgb24: &Tensor<R>,
    out_dtype: DType,
    float_ok: F,
    op: &'static str,
) -> Result<[usize; 2]> {
    let size = rgb24_frame_size(rgb24, "rgb24", op)?;
    check_float(out_dtype, float_ok, op)?;
    Ok(size)
}

/// Check an `rgb24_to_float_into` call and return `[H, W]`.
///
/// # Errors
///
/// Every error of [`plan_unpack`], plus the `out` errors of `validate_out`.
pub fn plan_unpack_into<R: Runtime<DType = DType>, F: Fn(DType) -> bool>(
    out: &Tensor<R>,
    rgb24: &Tensor<R>,
    layout: ConvLayout,
    float_ok: F,
    op: &'static str,
) -> Result<[usize; 2]> {
    let [h, w] = plan_unpack(rgb24, out.dtype(), float_ok, op)?;
    validate_out(out, &rgb_shape(layout, h, w), &[(rgb24, "rgb24")], op)?;
    Ok([h, w])
}

/// Check a `float_to_rgb24` call and return `[H, W]`.
///
/// # Errors
///
/// `Error::InvalidArgument` for an `rgb` that is not one frame, and
/// `Error::UnsupportedDType` for its dtype.
pub fn plan_pack<R: Runtime<DType = DType>, F: Fn(DType) -> bool>(
    rgb: &Tensor<R>,
    layout: ConvLayout,
    float_ok: F,
    op: &'static str,
) -> Result<[usize; 2]> {
    let size = rgb_frame_size(rgb, layout, "rgb", op)?;
    check_float(rgb.dtype(), float_ok, op)?;
    Ok(size)
}

/// Check a `float_to_rgb24_into` call and return `[H, W]`.
///
/// # Errors
///
/// Every error of [`plan_pack`], `Error::InvalidArgument` for an `out` that
/// is not U8, and the `out` errors of `validate_out`.
pub fn plan_pack_into<R: Runtime<DType = DType>, F: Fn(DType) -> bool>(
    out: &Tensor<R>,
    rgb: &Tensor<R>,
    layout: ConvLayout,
    float_ok: F,
    op: &'static str,
) -> Result<[usize; 2]> {
    let [h, w] = plan_pack(rgb, layout, float_ok, op)?;
    if out.dtype() != DType::U8 {
        return Err(Error::InvalidArgument {
            arg: "out",
            reason: format!(
                "{op}: out has dtype {:?}, expected U8; pass a U8 [H, W, 3] view",
                out.dtype()
            ),
        });
    }
    validate_out(out, &[h, w, 3], &[(rgb, "rgb")], op)?;
    Ok([h, w])
}
