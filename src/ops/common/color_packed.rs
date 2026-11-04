//! Checks and sample views of the packed `ColorOps` forms.
//!
//! A packed plane or RGB24 frame is a [`PackedPlane`]: U32 words `[rows, n]`
//! with column stride 1. On CPU and CUDA its samples are read in place as
//! U8 or U16 through a [`PlaneView`] in samples. WebGPU addresses the words.

use super::color_plan::PlaneView;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::traits::{PackedPlane, YuvPlanes};
use crate::runtime::Runtime;

/// Bytes of one RGB24 column.
pub const RGB24_COL_BYTES: usize = 3;

/// Check that `p` holds U32 words `[rows, n]` with column stride 1.
fn check_words<R: Runtime<DType = DType>>(
    p: &PackedPlane<'_, R>,
    name: &str,
    arg: &'static str,
    op: &'static str,
) -> Result<()> {
    let t = p.words;
    if t.dtype() != DType::U32 || t.shape().len() != 2 {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "{op}: {name} words are {:?} {:?}, expected U32 [rows, words]; pass the data \
                 as packed U32 words",
                t.dtype(),
                t.shape()
            ),
        });
    }
    if t.shape()[1] > 1 && t.strides()[1] != 1 {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "{op}: {name} words have column stride {}, expected 1; pass a view whose \
                 words are contiguous along each row",
                t.strides()[1]
            ),
        });
    }
    Ok(())
}

/// Check that the checked words of `p` hold `rows` rows of `width` columns
/// of `col_bytes` bytes.
fn check_extent<R: Runtime<DType = DType>>(
    p: &PackedPlane<'_, R>,
    name: &str,
    [rows, width, col_bytes]: [usize; 3],
    arg: &'static str,
    op: &'static str,
) -> Result<()> {
    let shape = p.words.shape();
    if shape[0] != rows || p.width != width {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "{op}: {name} has {} rows of {} columns, expected {rows} rows of {width} \
                 columns for the luma plane's frame; pass the {name} of the same frame",
                shape[0], p.width
            ),
        });
    }
    let need = PackedPlane::<R>::row_words(width * col_bytes);
    if shape[1] < need {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "{op}: {name} rows hold {} words, but {width} columns of {col_bytes} bytes \
                 need {need}; pass a words view of at least {need} columns",
                shape[1]
            ),
        });
    }
    Ok(())
}

/// Check the packed planes of one 4:2:0 frame and return the luma size
/// `[H, W]`.
///
/// # Errors
///
/// `Error::InvalidArgument` for argument `planes` when a plane is not U32
/// `[rows, words]` with column stride 1, when H or W is odd, or when a
/// plane has the wrong rows or width or holds too few words per row.
pub fn validate_packed_planes<R: Runtime<DType = DType>>(
    planes: &YuvPlanes<'_, R>,
    op: &'static str,
) -> Result<[usize; 2]> {
    let format = planes.format_name();
    let bytes = planes.sample_bytes();
    let (y, c0, c1) = match *planes {
        YuvPlanes::Nv12Packed { y, uv } | YuvPlanes::P010Packed { y, uv } => {
            (y, (uv, "plane uv"), None)
        }
        YuvPlanes::I420Packed { y, u, v } => (y, (u, "plane u"), Some((v, "plane v"))),
        _ => {
            return Err(Error::Internal(format!(
                "{op}: validate_packed_planes received unpacked {format} planes"
            )));
        }
    };
    let named = format!("{format} plane y");
    check_words(&y, &named, "planes", op)?;
    let (h, w) = (y.words.shape()[0], y.width);
    for (name, size) in [("H", h), ("W", w)] {
        if size % 2 != 0 {
            return Err(Error::InvalidArgument {
                arg: "planes",
                reason: format!(
                    "{op}: {format} luma {name} = {size} (packed plane y) is odd, but 4:2:0 \
                     needs an even H and W. Pad luma to an even size with pad_packed_luma first"
                ),
            });
        }
    }
    check_extent(&y, &named, [h, w, bytes], "planes", op)?;
    let chroma_width = if c1.is_some() { w / 2 } else { w };
    for (plane, name) in std::iter::once(c0).chain(c1) {
        let named = format!("{format} {name}");
        check_words(&plane, &named, "planes", op)?;
        check_extent(&plane, &named, [h / 2, chroma_width, bytes], "planes", op)?;
    }
    Ok([h, w])
}

/// Check that `p` is a packed RGB24 frame and return `[H, W]`.
///
/// # Errors
///
/// `Error::InvalidArgument` for argument `arg` when the words are not U32
/// `[H, words]` with column stride 1, or hold fewer than `ceil(3·W / 4)`
/// words per row.
pub fn packed_rgb24_frame_size<R: Runtime<DType = DType>>(
    p: &PackedPlane<'_, R>,
    arg: &'static str,
    op: &'static str,
) -> Result<[usize; 2]> {
    let name = format!("packed RGB24 {arg}");
    check_words(p, &name, arg, op)?;
    let rows = p.words.shape()[0];
    check_extent(p, &name, [rows, p.width, RGB24_COL_BYTES], arg, op)?;
    Ok([rows, p.width])
}

/// Check that `p` is a packed plane of `sample_bytes`-byte samples and return
/// `[H, W]`.
///
/// # Errors
///
/// `Error::InvalidArgument` for argument `arg` when the words are not U32
/// `[H, words]` with column stride 1, or hold too few words per row.
pub fn packed_plane_size<R: Runtime<DType = DType>>(
    p: &PackedPlane<'_, R>,
    sample_bytes: usize,
    arg: &'static str,
    op: &'static str,
) -> Result<[usize; 2]> {
    let name = format!("packed plane {arg}");
    check_words(p, &name, arg, op)?;
    let rows = p.words.shape()[0];
    check_extent(p, &name, [rows, p.width, sample_bytes], arg, op)?;
    Ok([rows, p.width])
}

/// Row stride of `p` in units of `sample_bytes`-byte samples.
fn row_stride_in<R: Runtime<DType = DType>>(
    p: &PackedPlane<'_, R>,
    sample_bytes: usize,
    op: &'static str,
) -> Result<isize> {
    let rs = p.words.strides()[0];
    let per_word = (4 / sample_bytes) as isize;
    rs.checked_mul(per_word)
        .ok_or_else(|| Error::InvalidArgument {
            arg: "planes",
            reason: format!(
                "{op}: packed row stride {rs} words overflows in units of {sample_bytes}-byte \
             samples; pass a view with a smaller row stride"
            ),
        })
}

/// Sample view of a checked packed plane of `sample_bytes`-byte samples.
///
/// # Errors
///
/// `Error::InvalidArgument` for argument `planes` when the row stride
/// overflows in samples.
pub fn packed_view<R: Runtime<DType = DType>>(
    p: &PackedPlane<'_, R>,
    sample_bytes: usize,
    op: &'static str,
) -> Result<PlaneView> {
    Ok(PlaneView {
        addr: p.words.ptr(),
        row_stride: row_stride_in(p, sample_bytes, op)?,
        col_stride: 1,
    })
}

/// U and V sample views of a checked packed interleaved chroma plane.
///
/// # Errors
///
/// As [`packed_view`].
pub fn packed_interleaved_views<R: Runtime<DType = DType>>(
    uv: &PackedPlane<'_, R>,
    sample_bytes: usize,
    op: &'static str,
) -> Result<(PlaneView, PlaneView)> {
    let base = packed_view(uv, sample_bytes, op)?;
    let u = PlaneView {
        col_stride: 2,
        ..base
    };
    let v = PlaneView {
        addr: base.addr + sample_bytes as u64,
        ..u
    };
    Ok((u, v))
}

/// `[C, H, W]` byte strides of a checked packed RGB24 frame.
///
/// # Errors
///
/// `Error::InvalidArgument` for argument `arg` when the row stride overflows
/// in bytes.
pub fn packed_rgb24_strides<R: Runtime<DType = DType>>(
    p: &PackedPlane<'_, R>,
    arg: &'static str,
    op: &'static str,
) -> Result<[isize; 3]> {
    let rs = p.words.strides()[0];
    let row = rs.checked_mul(4).ok_or_else(|| Error::InvalidArgument {
        arg,
        reason: format!(
            "{op}: packed RGB24 row stride {rs} words overflows in bytes; pass a view with a \
             smaller row stride"
        ),
    })?;
    Ok([1, row, RGB24_COL_BYTES as isize])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    fn words(shape: &[usize]) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::zeros(shape, DType::U32, &CpuDevice::new()).unwrap()
    }

    fn reason(result: Result<[usize; 2]>) -> String {
        match result {
            Err(Error::InvalidArgument { reason, .. }) => reason,
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn packed_nv12_accepts_partial_words_and_pitch() {
        let (y, uv) = (words(&[4, 3]), words(&[2, 2]));
        let planes = YuvPlanes::Nv12Packed {
            y: PackedPlane::new(&y, 6),
            uv: PackedPlane::new(&uv, 6),
        };
        assert_eq!(validate_packed_planes(&planes, "t").unwrap(), [4, 6]);
    }

    #[test]
    fn packed_planes_reject_short_rows_and_strided_words() {
        let (y, uv) = (words(&[4, 1]), words(&[2, 2]));
        let planes = YuvPlanes::Nv12Packed {
            y: PackedPlane::new(&y, 6),
            uv: PackedPlane::new(&uv, 6),
        };
        assert!(reason(validate_packed_planes(&planes, "t")).contains("need 2"));
        let strided = words(&[4, 4]).transpose(0, 1).unwrap();
        let p010 = YuvPlanes::P010Packed {
            y: PackedPlane::new(&strided, 2),
            uv: PackedPlane::new(&uv, 2),
        };
        assert!(reason(validate_packed_planes(&p010, "t")).contains("column stride"));
    }

    #[test]
    fn packed_views_count_samples() {
        let uv = words(&[2, 4]);
        let p = PackedPlane::new(&uv, 4);
        let (u, v) = packed_interleaved_views(&p, 2, "t").unwrap();
        assert_eq!((u.row_stride, u.col_stride), (8, 2));
        assert_eq!(v.addr, u.addr + 2);
        let rgb = words(&[3, 5]);
        let p = PackedPlane::new(&rgb, 6);
        assert_eq!(packed_rgb24_frame_size(&p, "rgb24", "t").unwrap(), [3, 6]);
        assert_eq!(packed_rgb24_strides(&p, "rgb24", "t").unwrap(), [1, 20, 3]);
    }

    #[test]
    fn words_shape_needs_a_word_pitch() {
        assert_eq!(
            PackedPlane::<CpuRuntime>::words_shape(2, 12).unwrap(),
            [2, 3]
        );
        match PackedPlane::<CpuRuntime>::words_shape(2, 10) {
            Err(Error::InvalidArgument { arg, reason }) => {
                assert_eq!(arg, "pitch_bytes");
                assert!(reason.contains("10 bytes") && reason.contains("12"));
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }
}
