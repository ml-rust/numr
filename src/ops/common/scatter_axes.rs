//! Axis collapsing for the `scatter_reduce` key kernels.
//!
//! A key kernel walks each source position's coordinates and recombines them
//! with the destination strides. Fewer axes mean fewer divisions per element,
//! and equal source and destination shapes collapse to at most three axes.

/// One collapsed axis: its source and destination extents, and the
/// destination stride of its innermost original axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScatterAxis {
    /// Source elements along this axis.
    pub src_extent: usize,
    /// Destination elements along this axis.
    pub dst_extent: usize,
    /// Destination stride of one step along this axis.
    pub dst_stride: usize,
    /// True for the scatter axis.
    pub is_dim: bool,
}

/// Merges adjacent axes a key kernel can walk as one.
///
/// An outer axis folds into the axis group inside it when neither is the
/// scatter axis and the inner group spans its full destination extent: the
/// source then walks the destination with one stride.
///
/// Returns the axes outermost first and the position of the scatter axis.
/// Both shapes must have the same rank, and `dim` must be below it.
pub fn collapse_scatter_axes(
    src_shape: &[usize],
    dst_shape: &[usize],
    dim: usize,
) -> (Vec<ScatterAxis>, usize) {
    let ndim = dst_shape.len();
    let mut strides = vec![1usize; ndim];
    for a in (0..ndim.saturating_sub(1)).rev() {
        strides[a] = strides[a + 1] * dst_shape[a + 1];
    }

    let mut groups: Vec<ScatterAxis> = Vec::with_capacity(ndim);
    for a in (0..ndim).rev() {
        let axis = ScatterAxis {
            src_extent: src_shape[a],
            dst_extent: dst_shape[a],
            dst_stride: strides[a],
            is_dim: a == dim,
        };
        match groups.last_mut() {
            Some(inner)
                if !axis.is_dim && !inner.is_dim && inner.src_extent == inner.dst_extent =>
            {
                inner.src_extent *= axis.src_extent;
                inner.dst_extent *= axis.dst_extent;
            }
            _ => groups.push(axis),
        }
    }
    groups.reverse();
    let dim_pos = groups.iter().position(|g| g.is_dim).unwrap_or(0);
    (groups, dim_pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_shapes_collapse_to_outer_dim_inner() {
        let (axes, dim_pos) = collapse_scatter_axes(&[2, 3, 4, 5, 6], &[2, 3, 9, 5, 6], 2);
        assert_eq!(dim_pos, 1);
        let ext: Vec<usize> = axes.iter().map(|a| a.src_extent).collect();
        let strides: Vec<usize> = axes.iter().map(|a| a.dst_stride).collect();
        assert_eq!(ext, vec![6, 4, 30]);
        assert_eq!(strides, vec![9 * 30, 30, 1]);
    }

    #[test]
    fn a_short_inner_axis_stays_separate() {
        // Axis 2 is shorter in the source, so axis 1 cannot fold into it.
        let (axes, dim_pos) = collapse_scatter_axes(&[4, 2, 3], &[5, 2, 7], 0);
        assert_eq!(dim_pos, 0);
        let ext: Vec<usize> = axes.iter().map(|a| a.src_extent).collect();
        let strides: Vec<usize> = axes.iter().map(|a| a.dst_stride).collect();
        assert_eq!(ext, vec![4, 2, 3]);
        assert_eq!(strides, vec![14, 7, 1]);
    }

    #[test]
    fn one_axis_stays_one_axis() {
        let (axes, dim_pos) = collapse_scatter_axes(&[10], &[3], 0);
        assert_eq!(dim_pos, 0);
        assert_eq!(axes.len(), 1);
        assert_eq!(axes[0].dst_stride, 1);
    }
}
