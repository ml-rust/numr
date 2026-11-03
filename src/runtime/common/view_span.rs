//! Element spans of strided views and the overlap rule for destination-passing ops.
//!
//! A view touches storage elements between a lowest and a highest index. Two
//! views of one storage cannot overlap when those spans are disjoint. The
//! test is conservative: interleaved views, such as two channel slices of one
//! NHWC buffer, have intersecting spans and count as overlapping.
//!
//! [`views_disjoint_along_one_dim`] is the exact test for that interleaved
//! case. [`check_out_disjoint_from`] runs it first and falls back to the span
//! test.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Inclusive range of storage element indices a strided view touches.
///
/// The range is `[offset + sum(min(0, (dim - 1) * stride)),
/// offset + sum(max(0, (dim - 1) * stride))]`. Returns `None` when the view
/// has no elements.
pub(crate) fn view_element_span(
    shape: &[usize],
    strides: &[isize],
    offset: usize,
) -> Option<(isize, isize)> {
    if shape.contains(&0) {
        return None;
    }
    let base = isize::try_from(offset).unwrap_or(isize::MAX);
    let (mut low, mut high) = (base, base);
    for (&dim, &stride) in shape.iter().zip(strides) {
        let reach = isize::try_from(dim - 1)
            .unwrap_or(isize::MAX)
            .saturating_mul(stride);
        if reach < 0 {
            low = low.saturating_add(reach);
        } else {
            high = high.saturating_add(reach);
        }
    }
    Some((low, high))
}

/// Whether two inclusive spans share at least one index.
pub(crate) fn spans_intersect(a: (isize, isize), b: (isize, isize)) -> bool {
    a.0 <= b.1 && b.0 <= a.1
}

/// Whether two tensors are the same view of one storage.
///
/// Strides of size-1 dimensions do not count, because they address nothing.
pub(crate) fn is_identical_view<R: Runtime<DType = DType>>(a: &Tensor<R>, b: &Tensor<R>) -> bool {
    a.shares_storage_with(b)
        && a.offset() == b.offset()
        && a.shape() == b.shape()
        && a.shape()
            .iter()
            .zip(a.strides().iter().zip(b.strides()))
            .all(|(&dim, (&sa, &sb))| dim == 1 || sa == sb)
}

/// Whether a view maps two logical indices to one storage element.
///
/// Only the broadcast case is detected: a stride of 0 on a dimension larger
/// than 1.
pub(crate) fn has_broadcast_dim(shape: &[usize], strides: &[isize]) -> bool {
    shape
        .iter()
        .zip(strides)
        .any(|(&dim, &stride)| dim > 1 && stride == 0)
}

/// Reject `out` and `src` when they share storage and their spans intersect.
///
/// Tensors on different storages always pass. An identical view also fails
/// here, so callers handle that case first with [`is_identical_view`].
///
/// # Errors
///
/// Returns `Error::InvalidArgument` for argument `out` on overlap.
pub(crate) fn check_views_disjoint<R: Runtime<DType = DType>>(
    out: &Tensor<R>,
    src: &Tensor<R>,
    op_name: &'static str,
) -> Result<()> {
    if let Some((a, b)) = intersecting_spans(out, src) {
        return Err(Error::InvalidArgument {
            arg: "out",
            reason: format!(
                "{op_name}: destination elements {}..={} overlap source elements {}..={} \
                 of the same storage",
                a.0, a.1, b.0, b.1
            ),
        });
    }
    Ok(())
}

/// Reject `out` when it can share an element with the read operand `src`.
///
/// Tensors on different storages always pass. Views of one storage pass when
/// [`views_disjoint_along_one_dim`] proves them disjoint, or else when their
/// element spans do not intersect. An identical view fails.
///
/// # Errors
///
/// Returns `Error::InvalidArgument` for argument `out` on overlap. The reason
/// names `src_name`.
pub(crate) fn check_out_disjoint_from<R: Runtime<DType = DType>>(
    out: &Tensor<R>,
    src: &Tensor<R>,
    src_name: &'static str,
    op_name: &'static str,
) -> Result<()> {
    if !out.shares_storage_with(src)
        || views_disjoint_along_one_dim(
            (out.shape(), out.strides(), out.offset()),
            (src.shape(), src.strides(), src.offset()),
        )
    {
        return Ok(());
    }
    if let Some((a, b)) = intersecting_spans(out, src) {
        return Err(Error::InvalidArgument {
            arg: "out",
            reason: format!(
                "{op_name}: out overlaps {src_name}: destination elements {}..={} and \
                 {src_name} elements {}..={} of the same storage intersect",
                a.0, a.1, b.0, b.1
            ),
        });
    }
    Ok(())
}

/// Spans of `out` and `src` when both share storage and the spans intersect.
fn intersecting_spans<R: Runtime<DType = DType>>(
    out: &Tensor<R>,
    src: &Tensor<R>,
) -> Option<((isize, isize), (isize, isize))> {
    if !out.shares_storage_with(src) {
        return None;
    }
    let a = view_element_span(out.shape(), out.strides(), out.offset())?;
    let b = view_element_span(src.shape(), src.strides(), src.offset())?;
    spans_intersect(a, b).then_some((a, b))
}

/// One strided view as `(shape, strides, offset)`.
pub(crate) type ViewGeometry<'a> = (&'a [usize], &'a [isize], usize);

/// Whether two views of one storage are provably disjoint along one dimension.
///
/// The test passes when all of these hold:
/// - Both views have one rank and equal strides on every dimension wider than 1.
/// - The shapes differ along at most one dimension `d`. With equal shapes,
///   any dimension can serve as `d`.
/// - The offset difference is `shift * stride[d]` for a whole `shift`.
/// - Index ranges `[0, a[d])` and `[shift, shift + b[d])` do not intersect.
/// - The combined view, which spans both ranges along `d`, addresses no
///   element twice. See [`layout_is_injective`].
///
/// An empty view touches nothing and always passes. `false` means "not
/// proven", never "overlapping".
pub(crate) fn views_disjoint_along_one_dim(a: ViewGeometry<'_>, b: ViewGeometry<'_>) -> bool {
    let (a_shape, a_strides, a_offset) = a;
    let (b_shape, b_strides, b_offset) = b;
    if a_shape.contains(&0) || b_shape.contains(&0) {
        return true;
    }
    let rank = a_shape.len();
    if b_shape.len() != rank || a_strides.len() != rank || b_strides.len() != rank {
        return false;
    }
    let strides_agree =
        (0..rank).all(|i| (a_shape[i] == 1 && b_shape[i] == 1) || a_strides[i] == b_strides[i]);
    if !strides_agree {
        return false;
    }
    let (Ok(a_off), Ok(b_off)) = (isize::try_from(a_offset), isize::try_from(b_offset)) else {
        return false;
    };
    let Some(diff) = b_off.checked_sub(a_off) else {
        return false;
    };
    let mut differing = (0..rank).filter(|&i| a_shape[i] != b_shape[i]);
    match (differing.next(), differing.next()) {
        (Some(_), Some(_)) => false,
        (Some(d), None) => disjoint_along(a_shape, a_strides, b_shape[d], diff, d),
        (None, _) => (0..rank).any(|d| disjoint_along(a_shape, a_strides, b_shape[d], diff, d)),
    }
}

/// The single-dimension test of [`views_disjoint_along_one_dim`] for one `d`.
///
/// `a` is described by `a_shape` and `a_strides`. `b` equals `a` except that
/// its length along `d` is `b_len` and its offset is `diff` elements past `a`'s.
fn disjoint_along(
    a_shape: &[usize],
    a_strides: &[isize],
    b_len: usize,
    diff: isize,
    d: usize,
) -> bool {
    let stride = a_strides[d];
    if diff.checked_rem(stride) != Some(0) {
        return false;
    }
    let Some(shift) = diff.checked_div(stride) else {
        return false;
    };
    let (Ok(a_len), Ok(b_len)) = (isize::try_from(a_shape[d]), isize::try_from(b_len)) else {
        return false;
    };
    let Some(b_end) = shift.checked_add(b_len) else {
        return false;
    };
    if shift < a_len && b_end > 0 {
        return false;
    }
    let low = shift.min(0);
    let high = a_len.max(b_end);
    let Some(Ok(union_len)) = high.checked_sub(low).map(usize::try_from) else {
        return false;
    };
    let mut union_shape = a_shape.to_vec();
    union_shape[d] = union_len;
    layout_is_injective(&union_shape, a_strides)
}

/// Whether a strided layout maps every logical index to its own element.
///
/// Sufficient test: order the dimensions wider than 1 by `|stride|`. Each
/// `|stride|` must exceed the farthest reach of the smaller dimensions, which
/// is the sum of `(dim - 1) * |stride|` over them. Arithmetic overflow
/// returns `false`.
pub(crate) fn layout_is_injective(shape: &[usize], strides: &[isize]) -> bool {
    let mut dims: Vec<(usize, usize)> = shape
        .iter()
        .zip(strides)
        .filter(|&(&dim, _)| dim > 1)
        .map(|(&dim, &stride)| (stride.unsigned_abs(), dim))
        .collect();
    dims.sort_unstable();
    let mut reach = 0usize;
    for (stride, dim) in dims {
        if stride <= reach {
            return false;
        }
        let Some(next) = stride
            .checked_mul(dim - 1)
            .and_then(|span| reach.checked_add(span))
        else {
            return false;
        };
        reach = next;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn span_of_contiguous_view() {
        assert_eq!(view_element_span(&[2, 3], &[3, 1], 4), Some((4, 9)));
    }

    #[test]
    fn span_of_strided_slice() {
        // narrow(-1, 2, 3) of a [2, 3, 4, 8] buffer: strides [96, 32, 8, 1], offset 2.
        let span = view_element_span(&[2, 3, 4, 3], &[96, 32, 8, 1], 2);
        assert_eq!(span, Some((2, 2 + 96 + 64 + 24 + 2)));
    }

    #[test]
    fn span_with_negative_stride() {
        // A flipped row: offset 4 points at the last element, stride -1.
        assert_eq!(view_element_span(&[5], &[-1], 4), Some((0, 4)));
        assert_eq!(view_element_span(&[2, 3], &[3, -1], 2), Some((0, 5)));
    }

    #[test]
    fn span_of_empty_view_is_none() {
        assert_eq!(view_element_span(&[2, 0], &[0, 1], 0), None);
    }

    #[test]
    fn span_of_scalar_is_one_element() {
        assert_eq!(view_element_span(&[], &[], 7), Some((7, 7)));
    }

    #[test]
    fn disjoint_spans_do_not_intersect() {
        assert!(!spans_intersect((0, 3), (4, 7)));
        assert!(!spans_intersect((4, 7), (0, 3)));
    }

    #[test]
    fn touching_spans_intersect() {
        assert!(spans_intersect((0, 4), (4, 7)));
        assert!(spans_intersect((0, 9), (3, 4)));
    }

    #[test]
    fn interleaved_channel_slices_intersect() {
        // Channels 0..2 and 2..4 of a [4, 4] buffer: disjoint elements, but
        // the spans interleave, so the conservative rule reports overlap.
        let first = view_element_span(&[4, 2], &[4, 1], 0);
        let second = view_element_span(&[4, 2], &[4, 1], 2);
        assert!(spans_intersect(first.unwrap(), second.unwrap()));
    }

    #[test]
    fn broadcast_dimension_is_detected() {
        assert!(has_broadcast_dim(&[3, 4], &[0, 1]));
        assert!(!has_broadcast_dim(&[1, 4], &[0, 1]));
        assert!(!has_broadcast_dim(&[3, 4], &[4, 1]));
    }

    fn buffer() -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::zeros(&[2, 3, 4, 8], DType::F32, &CpuDevice::new()).unwrap()
    }

    #[test]
    fn identical_view_is_recognised() {
        let buf = buffer();
        let a = buf.narrow(-1, 2, 3).unwrap();
        let b = buf.narrow(-1, 2, 3).unwrap();
        let c = buf.narrow(-1, 3, 3).unwrap();
        assert!(is_identical_view(&a, &b));
        assert!(!is_identical_view(&a, &c));
    }

    #[test]
    fn views_of_one_buffer_overlap() {
        let buf = buffer();
        let a = buf.narrow(-1, 0, 2).unwrap();
        let b = buf.narrow(-1, 4, 2).unwrap();
        let err = check_views_disjoint(&a, &b, "copy_into").unwrap_err();
        assert!(matches!(err, Error::InvalidArgument { arg: "out", .. }));
    }

    #[test]
    fn leading_halves_of_one_buffer_are_disjoint() {
        let buf = buffer();
        let a = buf.narrow(0, 0, 1).unwrap();
        let b = buf.narrow(0, 1, 1).unwrap();
        assert!(check_views_disjoint(&a, &b, "copy_into").is_ok());
    }

    fn geometry(t: &Tensor<CpuRuntime>) -> ViewGeometry<'_> {
        (t.shape(), t.strides(), t.offset())
    }

    #[test]
    fn disjoint_channel_slices_are_proven_disjoint() {
        let buf = buffer();
        let a = buf.narrow(-1, 0, 2).unwrap();
        let b = buf.narrow(-1, 4, 2).unwrap();
        assert!(views_disjoint_along_one_dim(geometry(&a), geometry(&b)));
        assert!(views_disjoint_along_one_dim(geometry(&b), geometry(&a)));
    }

    #[test]
    fn adjacent_dense_block_slices_are_proven_disjoint() {
        // Read prefix [0, 6), write [6, 8): the RRDB dense-block pattern.
        let buf = buffer();
        let read = buf.narrow(-1, 0, 6).unwrap();
        let write = buf.narrow(-1, 6, 2).unwrap();
        assert!(views_disjoint_along_one_dim(
            geometry(&write),
            geometry(&read)
        ));
    }

    #[test]
    fn nchw_channel_slices_are_proven_disjoint() {
        let buf =
            Tensor::<CpuRuntime>::zeros(&[1, 10, 5, 6], DType::F32, &CpuDevice::new()).unwrap();
        let read = buf.narrow(1, 0, 4).unwrap();
        let write = buf.narrow(1, 4, 2).unwrap();
        assert!(views_disjoint_along_one_dim(
            geometry(&write),
            geometry(&read)
        ));
    }

    #[test]
    fn overlapping_channel_slices_are_not_proven_disjoint() {
        let buf = buffer();
        let a = buf.narrow(-1, 0, 4).unwrap();
        let b = buf.narrow(-1, 2, 4).unwrap();
        assert!(!views_disjoint_along_one_dim(geometry(&a), geometry(&b)));
    }

    #[test]
    fn identical_view_is_not_proven_disjoint() {
        let buf = buffer();
        let a = buf.narrow(-1, 2, 3).unwrap();
        assert!(!views_disjoint_along_one_dim(geometry(&a), geometry(&a)));
    }

    #[test]
    fn different_strides_are_not_proven_disjoint() {
        // A transposed view of the same storage: strides differ, so the exact
        // rule declines and the span fallback decides.
        let buf = Tensor::<CpuRuntime>::zeros(&[4, 4], DType::F32, &CpuDevice::new()).unwrap();
        let a = buf.narrow(1, 0, 2).unwrap();
        let t = buf.transpose(0, 1).unwrap();
        let b = t.narrow(1, 2, 2).unwrap();
        assert!(!views_disjoint_along_one_dim(geometry(&a), geometry(&b)));
    }

    #[test]
    fn shift_past_the_row_is_not_proven_disjoint() {
        // Channels [0, 2) and a view 12 elements on: its "channels" 12..14 of
        // row w are channels 4..6 of row w + 1. The combined view repeats
        // elements, so the exact rule declines.
        let shape: [usize; 4] = [2, 3, 4, 2];
        let strides: [isize; 4] = [96, 32, 8, 1];
        assert!(!views_disjoint_along_one_dim(
            (&shape, &strides, 0),
            (&shape, &strides, 12)
        ));
    }

    #[test]
    fn shapes_differing_on_two_dims_are_not_proven_disjoint() {
        let strides: [isize; 4] = [96, 32, 8, 1];
        assert!(!views_disjoint_along_one_dim(
            (&[2, 3, 4, 2], &strides, 0),
            (&[2, 3, 2, 3], &strides, 4)
        ));
    }

    #[test]
    fn empty_view_is_disjoint() {
        let strides: [isize; 2] = [8, 1];
        assert!(views_disjoint_along_one_dim(
            (&[4, 0], &strides, 0),
            (&[4, 8], &strides, 0)
        ));
    }

    #[test]
    fn injective_layouts() {
        assert!(layout_is_injective(&[2, 3, 4, 8], &[96, 32, 8, 1]));
        assert!(layout_is_injective(&[3, 7], &[7, -1]));
        assert!(!layout_is_injective(&[3, 4], &[0, 1]));
        assert!(!layout_is_injective(&[3, 4], &[2, 1]));
        assert!(layout_is_injective(&[1, 4], &[0, 1]));
    }

    #[test]
    fn out_check_accepts_disjoint_interleaved_slices() {
        let buf = buffer();
        let read = buf.narrow(-1, 0, 6).unwrap();
        let write = buf.narrow(-1, 6, 2).unwrap();
        assert!(check_out_disjoint_from(&write, &read, "input", "conv2d_fused").is_ok());
    }

    #[test]
    fn out_check_names_the_overlapping_operand() {
        let buf = buffer();
        let read = buf.narrow(-1, 0, 6).unwrap();
        let write = buf.narrow(-1, 4, 2).unwrap();
        let err = check_out_disjoint_from(&write, &read, "residual", "conv2d_fused").unwrap_err();
        match err {
            Error::InvalidArgument { arg, reason } => {
                assert_eq!(arg, "out");
                assert!(reason.contains("residual"), "{reason}");
            }
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn out_check_rejects_identical_view() {
        let buf = buffer();
        let a = buf.narrow(-1, 2, 3).unwrap();
        assert!(check_out_disjoint_from(&a, &a, "input", "conv2d_fused").is_err());
    }

    #[test]
    fn out_check_falls_back_to_spans() {
        // Batch 0 and the first half of batch 1's channels. Along the channel
        // dimension the shift runs past a row, so the exact rule declines.
        // The spans do not intersect, so the fallback accepts.
        let buf = buffer();
        let a = buf.narrow(0, 0, 1).unwrap();
        let b = buf.narrow(0, 1, 1).unwrap().narrow(-1, 0, 4).unwrap();
        assert!(!views_disjoint_along_one_dim(geometry(&a), geometry(&b)));
        assert!(check_out_disjoint_from(&a, &b, "input", "conv2d_fused").is_ok());
    }

    #[test]
    fn separate_buffers_never_overlap() {
        let a = buffer();
        let b = buffer();
        assert!(check_views_disjoint(&a, &b, "copy_into").is_ok());
    }
}
