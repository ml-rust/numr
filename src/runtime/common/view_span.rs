//! Element spans of strided views and the overlap rule for destination-passing ops.
//!
//! A view touches storage elements between a lowest and a highest index. Two
//! views of one storage cannot overlap when those spans are disjoint. The
//! test is conservative: interleaved views, such as two channel slices of one
//! NHWC buffer, have intersecting spans and count as overlapping.

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
    if !out.shares_storage_with(src) {
        return Ok(());
    }
    let out_span = view_element_span(out.shape(), out.strides(), out.offset());
    let src_span = view_element_span(src.shape(), src.strides(), src.offset());
    if let (Some(a), Some(b)) = (out_span, src_span)
        && spans_intersect(a, b)
    {
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

    #[test]
    fn separate_buffers_never_overlap() {
        let a = buffer();
        let b = buffer();
        assert!(check_views_disjoint(&a, &b, "copy_into").is_ok());
    }
}
