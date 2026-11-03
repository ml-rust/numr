//! Shared argument checks of the `ColorOps` conversions.
//!
//! Planes, float RGB and RGB24 tensors are checked here, so every backend
//! rejects the same calls with the same errors.

use super::color_packed::validate_packed_planes;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::traits::{ConvLayout, YuvPlanes};
use crate::runtime::Runtime;
use crate::runtime::common::view_span::{
    layout_is_injective, spans_intersect, view_element_span, views_disjoint_along_one_dim,
};
use crate::tensor::Tensor;

/// Shape of one float RGB frame of `h x w` pixels in `layout`.
pub fn rgb_shape(layout: ConvLayout, h: usize, w: usize) -> [usize; 4] {
    match layout {
        ConvLayout::Nchw => [1, 3, h, w],
        ConvLayout::Nhwc => [1, h, w, 3],
    }
}

fn layout_name(layout: ConvLayout) -> &'static str {
    match layout {
        ConvLayout::Nchw => "[1, 3, H, W]",
        ConvLayout::Nhwc => "[1, H, W, 3]",
    }
}

fn check_plane<R: Runtime<DType = DType>>(
    t: &Tensor<R>,
    name: &str,
    format: &str,
    dtype: DType,
    want: [usize; 2],
    op: &'static str,
) -> Result<()> {
    if t.dtype() != dtype {
        return Err(Error::InvalidArgument {
            arg: "planes",
            reason: format!(
                "{op}: {format} plane {name} has dtype {:?}, expected {dtype:?}; \
                 pass a {dtype:?} view of the plane",
                t.dtype()
            ),
        });
    }
    if t.shape() != want {
        return Err(Error::InvalidArgument {
            arg: "planes",
            reason: format!(
                "{op}: {format} plane {name} has shape {:?}, expected {want:?} for the \
                 luma plane's frame; pass the {name} plane of the same frame",
                t.shape()
            ),
        });
    }
    Ok(())
}

/// Check the planes of one 4:2:0 frame and return the luma size `[H, W]`.
///
/// # Errors
///
/// `Error::InvalidArgument` for argument `planes` when a plane has the wrong
/// dtype, rank or shape, or when H or W is odd. Packed planes take the
/// checks of [`validate_packed_planes`].
pub fn validate_planes<R: Runtime<DType = DType>>(
    planes: &YuvPlanes<'_, R>,
    op: &'static str,
) -> Result<[usize; 2]> {
    if planes.is_packed() {
        return validate_packed_planes(planes, op);
    }
    let format = planes.format_name();
    let y = planes.y();
    let sample = match planes {
        YuvPlanes::P010 { .. } => DType::U16,
        _ => DType::U8,
    };
    if y.shape().len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "planes",
            reason: format!(
                "{op}: {format} plane y has shape {:?} of rank {}, expected [H, W]; \
                 pass a 2-D view of the luma plane",
                y.shape(),
                y.shape().len()
            ),
        });
    }
    let (h, w) = (y.shape()[0], y.shape()[1]);
    for (dim, name, size) in [(0, "H", h), (1, "W", w)] {
        if size % 2 != 0 {
            return Err(Error::InvalidArgument {
                arg: "planes",
                reason: format!(
                    "{op}: {format} luma {name} = {size} (dim {dim} of plane y) is odd, but \
                     4:2:0 needs an even H and W; pad the frame to an even size first"
                ),
            });
        }
    }
    check_plane(y, "y", format, sample, [h, w], op)?;
    match *planes {
        YuvPlanes::Nv12 { uv, .. } | YuvPlanes::P010 { uv, .. } => {
            check_plane(uv, "uv", format, sample, [h / 2, w], op)?;
        }
        YuvPlanes::I420 { u, v, .. } => {
            check_plane(u, "u", format, sample, [h / 2, w / 2], op)?;
            check_plane(v, "v", format, sample, [h / 2, w / 2], op)?;
        }
        // Packed planes returned from `validate_packed_planes` above.
        YuvPlanes::Nv12Packed { .. }
        | YuvPlanes::I420Packed { .. }
        | YuvPlanes::P010Packed { .. } => {}
    }
    Ok([h, w])
}

/// Check that `rgb` is a float RGB frame in `layout` and return `[H, W]`.
///
/// # Errors
///
/// `Error::InvalidArgument` for argument `arg` when the rank is not 4, the
/// batch is not 1 or the channel count is not 3.
pub fn rgb_frame_size<R: Runtime<DType = DType>>(
    rgb: &Tensor<R>,
    layout: ConvLayout,
    arg: &'static str,
    op: &'static str,
) -> Result<[usize; 2]> {
    let shape = rgb.shape();
    let dims = match (shape.len(), layout) {
        (4, ConvLayout::Nchw) => Some((shape[0], shape[1], shape[2], shape[3])),
        (4, ConvLayout::Nhwc) => Some((shape[0], shape[3], shape[1], shape[2])),
        _ => None,
    };
    match dims {
        Some((1, 3, h, w)) => Ok([h, w]),
        _ => Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "{op}: {arg} has shape {shape:?}, expected one RGB frame {}; \
                 pass one frame, or narrow a batch to one frame",
                layout_name(layout)
            ),
        }),
    }
}

/// Check that `rgb24` is U8 `[H, W, 3]` and return `[H, W]`.
///
/// # Errors
///
/// `Error::InvalidArgument` for argument `arg` on another dtype or shape.
pub fn rgb24_frame_size<R: Runtime<DType = DType>>(
    rgb24: &Tensor<R>,
    arg: &'static str,
    op: &'static str,
) -> Result<[usize; 2]> {
    let shape = rgb24.shape();
    if rgb24.dtype() != DType::U8 || shape.len() != 3 || shape[2] != 3 {
        return Err(Error::InvalidArgument {
            arg,
            reason: format!(
                "{op}: {arg} is {:?} {shape:?}, expected U8 [H, W, 3]; \
                 pass packed RGB24 as a U8 [H, W, 3] view",
                rgb24.dtype()
            ),
        });
    }
    Ok([shape[0], shape[1]])
}

/// Check that a destination view addresses each element once.
///
/// # Errors
///
/// `Error::InvalidArgument` for argument `arg` when two indices of `t` reach
/// one element.
pub fn check_writable<R: Runtime<DType = DType>>(
    t: &Tensor<R>,
    arg: &'static str,
    name: &str,
    op: &'static str,
) -> Result<()> {
    if layout_is_injective(t.shape(), t.strides()) {
        return Ok(());
    }
    Err(Error::InvalidArgument {
        arg,
        reason: format!(
            "{op}: {name} with shape {:?} and strides {:?} reaches some element twice; \
             pass a view that addresses each element once",
            t.shape(),
            t.strides()
        ),
    })
}

/// Inclusive byte range of storage a view touches, or `None` when empty.
fn byte_span<R: Runtime<DType = DType>>(t: &Tensor<R>) -> Option<(isize, isize)> {
    let (low, high) = view_element_span(t.shape(), t.strides(), t.offset())?;
    let size = t.dtype().size_in_bytes() as isize;
    Some((
        low.saturating_mul(size),
        high.saturating_mul(size).saturating_add(size - 1),
    ))
}

/// Reject a written view `dst` that can share a byte with `other`.
///
/// Views of different storages pass. Views of one storage and one dtype pass
/// when they are disjoint along one dimension. Otherwise their byte spans
/// must not intersect.
///
/// # Errors
///
/// `Error::InvalidArgument` for argument `arg` on overlap.
pub fn check_disjoint<R: Runtime<DType = DType>>(
    (dst, dst_name): (&Tensor<R>, &str),
    (other, other_name): (&Tensor<R>, &str),
    arg: &'static str,
    op: &'static str,
) -> Result<()> {
    if !dst.shares_storage_with(other) {
        return Ok(());
    }
    if dst.dtype() == other.dtype()
        && views_disjoint_along_one_dim(
            (dst.shape(), dst.strides(), dst.offset()),
            (other.shape(), other.strides(), other.offset()),
        )
    {
        return Ok(());
    }
    let (Some(a), Some(b)) = (byte_span(dst), byte_span(other)) else {
        return Ok(());
    };
    if !spans_intersect(a, b) {
        return Ok(());
    }
    Err(Error::InvalidArgument {
        arg,
        reason: format!(
            "{op}: {dst_name} overlaps {other_name}: bytes {}..={} and {}..={} of one storage \
             intersect; pass views that share no byte",
            a.0, a.1, b.0, b.1
        ),
    })
}

/// Named planes of `planes`, luma first.
pub fn named_planes<'a, R: Runtime>(
    planes: &YuvPlanes<'a, R>,
) -> Vec<(&'a Tensor<R>, &'static str)> {
    match *planes {
        YuvPlanes::Nv12 { y, uv } | YuvPlanes::P010 { y, uv } => {
            vec![(y, "plane y"), (uv, "plane uv")]
        }
        YuvPlanes::I420 { y, u, v } => vec![(y, "plane y"), (u, "plane u"), (v, "plane v")],
        YuvPlanes::Nv12Packed { y, uv } | YuvPlanes::P010Packed { y, uv } => {
            vec![(y.words, "plane y"), (uv.words, "plane uv")]
        }
        YuvPlanes::I420Packed { y, u, v } => vec![
            (y.words, "plane y"),
            (u.words, "plane u"),
            (v.words, "plane v"),
        ],
    }
}

/// Check the written planes of `rgb_to_yuv_into`: each addresses its
/// elements once, and no plane shares a byte with another plane or `rgb`.
///
/// # Errors
///
/// `Error::InvalidArgument` for argument `planes`.
pub fn validate_written_planes<R: Runtime<DType = DType>>(
    planes: &YuvPlanes<'_, R>,
    rgb: &Tensor<R>,
    op: &'static str,
) -> Result<()> {
    let named = named_planes(planes);
    for (i, &(plane, name)) in named.iter().enumerate() {
        check_writable(plane, "planes", name, op)?;
        check_disjoint((plane, name), (rgb, "rgb"), "planes", op)?;
        for &(other, other_name) in &named[i + 1..] {
            check_disjoint((plane, name), (other, other_name), "planes", op)?;
        }
    }
    Ok(())
}

/// Check a float destination `out` of shape `want` against its read
/// operands `reads`.
///
/// # Errors
///
/// `Error::ShapeMismatch` for another shape, and `Error::InvalidArgument` for
/// argument `out` when it repeats an element or overlaps a read operand.
pub fn validate_out<R: Runtime<DType = DType>>(
    out: &Tensor<R>,
    want: &[usize],
    reads: &[(&Tensor<R>, &str)],
    op: &'static str,
) -> Result<()> {
    if out.shape() != want {
        return Err(Error::shape_mismatch(want, out.shape()));
    }
    check_writable(out, "out", "out", op)?;
    for &(read, name) in reads {
        check_disjoint((out, "out"), (read, name), "out", op)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    fn zeros(shape: &[usize], dtype: DType) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::zeros(shape, dtype, &CpuDevice::new()).unwrap()
    }

    fn reason(result: Result<[usize; 2]>) -> String {
        match result {
            Err(Error::InvalidArgument {
                arg: "planes",
                reason,
            }) => reason,
            other => panic!("expected InvalidArgument for planes, got {other:?}"),
        }
    }

    #[test]
    fn planes_accept_nv12_and_reject_odd_sizes() {
        let (y, uv) = (zeros(&[4, 6], DType::U8), zeros(&[2, 6], DType::U8));
        let planes = YuvPlanes::Nv12 { y: &y, uv: &uv };
        assert_eq!(validate_planes(&planes, "t").unwrap(), [4, 6]);
        let odd = zeros(&[4, 5], DType::U8);
        let planes = YuvPlanes::Nv12 { y: &odd, uv: &uv };
        assert!(reason(validate_planes(&planes, "t")).contains("W = 5"));
        let odd = zeros(&[3, 6], DType::U8);
        let planes = YuvPlanes::Nv12 { y: &odd, uv: &uv };
        assert!(reason(validate_planes(&planes, "t")).contains("H = 3"));
    }

    #[test]
    fn planes_reject_wrong_dtype_and_chroma_shape() {
        let (y, uv) = (zeros(&[4, 6], DType::U8), zeros(&[2, 6], DType::U8));
        let planes = YuvPlanes::P010 { y: &y, uv: &uv };
        assert!(reason(validate_planes(&planes, "t")).contains("U16"));
        let (u, v) = (zeros(&[2, 3], DType::U8), zeros(&[2, 2], DType::U8));
        let planes = YuvPlanes::I420 {
            y: &y,
            u: &u,
            v: &v,
        };
        assert!(reason(validate_planes(&planes, "t")).contains("plane v"));
    }

    #[test]
    fn rgb_sizes_follow_the_layout() {
        let nchw = zeros(&[1, 3, 4, 6], DType::F32);
        assert_eq!(
            rgb_frame_size(&nchw, ConvLayout::Nchw, "rgb", "t").unwrap(),
            [4, 6]
        );
        let nhwc = zeros(&[1, 4, 6, 3], DType::F32);
        assert_eq!(
            rgb_frame_size(&nhwc, ConvLayout::Nhwc, "rgb", "t").unwrap(),
            [4, 6]
        );
        assert!(rgb_frame_size(&nchw, ConvLayout::Nhwc, "rgb", "t").is_err());
        let batch = zeros(&[2, 3, 4, 6], DType::F32);
        assert!(rgb_frame_size(&batch, ConvLayout::Nchw, "rgb", "t").is_err());
    }

    #[test]
    fn planes_of_one_surface_are_disjoint() {
        let surface = zeros(&[6, 8], DType::U8);
        let y = surface.narrow(0, 0, 4).unwrap().narrow(1, 0, 6).unwrap();
        let uv = surface.narrow(0, 4, 2).unwrap().narrow(1, 0, 6).unwrap();
        let rgb = zeros(&[1, 3, 4, 6], DType::F32);
        let planes = YuvPlanes::Nv12 { y: &y, uv: &uv };
        assert!(validate_written_planes(&planes, &rgb, "t").is_ok());
        let uv_overlap = surface.narrow(0, 3, 2).unwrap().narrow(1, 0, 6).unwrap();
        let planes = YuvPlanes::Nv12 {
            y: &y,
            uv: &uv_overlap,
        };
        assert!(validate_written_planes(&planes, &rgb, "t").is_err());
    }
}
