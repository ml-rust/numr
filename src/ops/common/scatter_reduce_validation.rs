//! Shared `scatter_reduce` shape validation.
//!
//! Source element `e` lands at its own coordinates, with the coordinate on the
//! scatter axis replaced by `index[e]`. Off that axis, a source coordinate is a
//! destination coordinate, so the source must not be longer than the
//! destination on any other axis. A longer axis would address past the
//! destination's end.

use crate::error::{Error, Result};

/// Checks that `src_shape` fits inside `dst_shape` on every axis but `dim`.
///
/// Both shapes must have the same rank; the caller checks that first.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] naming the first axis where the source
/// is longer than the destination.
pub fn validate_scatter_extents(
    dst_shape: &[usize],
    src_shape: &[usize],
    dim: usize,
) -> Result<()> {
    let longer = dst_shape
        .iter()
        .zip(src_shape)
        .enumerate()
        .find(|&(axis, (dst, src))| axis != dim && src > dst);
    match longer {
        Some((axis, (dst, src))) => Err(Error::InvalidArgument {
            arg: "src",
            reason: format!(
                "src shape {src_shape:?} is longer than dst shape {dst_shape:?} on axis {axis} \
                 ({src} > {dst}); only the scatter axis {dim} may differ that way"
            ),
        }),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scatter_axis_may_be_longer() {
        assert!(validate_scatter_extents(&[2, 3], &[2, 9], 1).is_ok());
    }

    #[test]
    fn a_shorter_source_axis_is_accepted() {
        assert!(validate_scatter_extents(&[4, 3], &[2, 3], 1).is_ok());
    }

    #[test]
    fn a_longer_source_axis_is_rejected() {
        let err = validate_scatter_extents(&[2, 3], &[5, 3], 1).unwrap_err();
        assert!(err.to_string().contains("axis 0"), "{err}");
    }
}
