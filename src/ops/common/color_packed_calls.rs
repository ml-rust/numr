//! Per-call checks of the packed RGB24 `ColorOps` methods, shared by every
//! backend.

use super::color_calls::{check_float, plan_pack};
use super::color_packed::{RGB24_COL_BYTES, packed_plane_size, packed_rgb24_frame_size};
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

/// A checked `pad_packed_luma` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedLumaPad {
    /// Source `[H, W]`, both at least 1.
    pub source: [usize; 2],
    /// Target `[rows, columns]`, each at least its source size.
    pub target: [usize; 2],
    /// Bytes per sample: 1 for U8, 2 for U16.
    pub sample_bytes: usize,
    /// Words per target row: `ceil(columns · sample_bytes / 4)`.
    pub row_words: usize,
}

impl PackedLumaPad {
    /// Shape `[rows, row_words]` of the new U32 words.
    pub fn out_shape(&self) -> [usize; 2] {
        [self.target[0], self.row_words]
    }
}

/// Check a `pad_packed_luma` call.
///
/// # Errors
///
/// `Error::InvalidArgument` for argument:
/// - `sample` when it is neither U8 nor U16.
/// - `y` when its words are malformed, or it has no rows or no columns.
/// - `rows` or `columns` when the target is below the source size, or the
///   target words overflow `usize`.
pub fn plan_pad_packed_luma<R: Runtime<DType = DType>>(
    y: &PackedPlane<'_, R>,
    rows: usize,
    columns: usize,
    sample: DType,
    op: &'static str,
) -> Result<PackedLumaPad> {
    let sample_bytes = match sample {
        DType::U8 => 1,
        DType::U16 => 2,
        other => {
            return Err(Error::InvalidArgument {
                arg: "sample",
                reason: format!(
                    "{op}: sample dtype {other:?} is not a luma sample type. Pass DType::U8 for \
                     NV12 and I420 or DType::U16 for P010"
                ),
            });
        }
    };
    let [h, w] = packed_plane_size(y, sample_bytes, "y", op)?;
    if h == 0 || w == 0 {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: format!(
                "{op}: luma plane is {h} rows of {w} columns, which has no edge to replicate. \
                 Pass a plane with at least one row and one column"
            ),
        });
    }
    if rows < h {
        return Err(Error::InvalidArgument {
            arg: "rows",
            reason: format!(
                "{op}: target {rows} rows is below the {h} rows of the luma plane. Pass at \
                 least {h} rows"
            ),
        });
    }
    if columns < w {
        return Err(Error::InvalidArgument {
            arg: "columns",
            reason: format!(
                "{op}: target {columns} columns is below the {w} columns of the luma plane. \
                 Pass at least {w} columns"
            ),
        });
    }
    let row_words = columns
        .checked_mul(sample_bytes)
        .map(PackedPlane::<R>::row_words)
        .filter(|words| words.checked_mul(rows).is_some())
        .ok_or_else(|| Error::InvalidArgument {
            arg: "columns",
            reason: format!(
                "{op}: {rows} rows of {columns} columns of {sample_bytes}-byte samples overflow \
                 the word count. Pass a smaller target"
            ),
        })?;
    Ok(PackedLumaPad {
        source: [h, w],
        target: [rows, columns],
        sample_bytes,
        row_words,
    })
}
