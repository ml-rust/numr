//! Shared argument checks and plans for the `ImageOps` data-movement ops.
//!
//! Every backend works in canonical `(N, C, H, W)` coordinates. A plan holds
//! the canonical input and output extents, the source map of the op, and the
//! output shape in the caller's layout. [`canonical_strides`] reorders a
//! tensor's strides into canonical order, so one kernel serves NCHW and NHWC.

use super::image_nearest::NearestAxis;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::traits::ConvLayout;
use crate::runtime::Runtime;
use crate::runtime::common::view_span::{check_out_disjoint_from, has_broadcast_dim};
use crate::tensor::Tensor;

/// Source map of one image op in canonical coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageMap {
    /// `in[n, c·r² + (h % r)·r + w % r, h / r, w / r]`.
    Shuffle {
        /// Upscale factor `r`.
        factor: usize,
    },
    /// `in[n, c / r², h·r + (c % r²) / r, w·r + c % r]`.
    Unshuffle {
        /// Downscale factor `r`.
        factor: usize,
    },
    /// `in[n, c, src_h(h), src_w(w)]`.
    Nearest {
        /// Source-index rule along H.
        h: NearestAxis,
        /// Source-index rule along W.
        w: NearestAxis,
    },
}

/// Validated geometry of one image op call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImagePlan {
    /// Operation name for errors.
    pub op: &'static str,
    /// Source map.
    pub map: ImageMap,
    /// Layout of input and output.
    pub layout: ConvLayout,
    /// Input extents as `[N, C, H, W]`.
    pub in_dims: [usize; 4],
    /// Output extents as `[N, C, H, W]`.
    pub out_dims: [usize; 4],
    /// Output shape in `layout` order.
    pub out_shape: [usize; 4],
}

impl ImagePlan {
    /// Output element count.
    pub fn numel(&self) -> usize {
        self.out_dims.iter().product()
    }

    /// True when kernels walk the output with C fastest (NHWC order).
    pub fn channels_last(&self) -> bool {
        self.layout == ConvLayout::Nhwc
    }
}

/// Reorder the strides of a 4-D tensor in `layout` into `[N, C, H, W]` order.
pub fn canonical_strides(strides: &[isize], layout: ConvLayout) -> [isize; 4] {
    match layout {
        ConvLayout::Nchw => [strides[0], strides[1], strides[2], strides[3]],
        ConvLayout::Nhwc => [strides[0], strides[3], strides[1], strides[2]],
    }
}

/// Canonical `[N, C, H, W]` extents of a 4-D input in `layout`.
pub(super) fn canonical_dims<R: Runtime<DType = DType>>(
    input: &Tensor<R>,
    layout: ConvLayout,
    op: &'static str,
) -> Result<[usize; 4]> {
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: format!(
                "{op}: input shape {shape:?} has rank {}, expected 4 ({}); reshape the input to 4-D",
                shape.len(),
                layout_name(layout)
            ),
        });
    }
    Ok(match layout {
        ConvLayout::Nchw => [shape[0], shape[1], shape[2], shape[3]],
        ConvLayout::Nhwc => [shape[0], shape[3], shape[1], shape[2]],
    })
}

/// Shape in `layout` order of canonical extents `dims`.
pub(super) fn layout_shape(dims: [usize; 4], layout: ConvLayout) -> [usize; 4] {
    match layout {
        ConvLayout::Nchw => dims,
        ConvLayout::Nhwc => [dims[0], dims[2], dims[3], dims[1]],
    }
}

fn layout_name(layout: ConvLayout) -> &'static str {
    match layout {
        ConvLayout::Nchw => "[N, C, H, W]",
        ConvLayout::Nhwc => "[N, H, W, C]",
    }
}

fn check_factor(factor: usize, op: &'static str) -> Result<()> {
    if factor == 0 {
        return Err(Error::InvalidArgument {
            arg: "factor",
            reason: format!("{op}: factor is 0; pass a factor of at least 1"),
        });
    }
    Ok(())
}

fn factor_squared(factor: usize, op: &'static str) -> Result<usize> {
    factor
        .checked_mul(factor)
        .ok_or_else(|| Error::InvalidArgument {
            arg: "factor",
            reason: format!("{op}: factor {factor} squared overflows usize; pass a smaller factor"),
        })
}

pub(super) fn plan(
    op: &'static str,
    map: ImageMap,
    layout: ConvLayout,
    in_dims: [usize; 4],
    out_dims: [usize; 4],
) -> ImagePlan {
    ImagePlan {
        op,
        map,
        layout,
        in_dims,
        out_dims,
        out_shape: layout_shape(out_dims, layout),
    }
}

/// Check `pixel_shuffle` arguments and build its plan.
///
/// # Errors
///
/// `Error::InvalidArgument` for a rank other than 4, a zero factor, a channel
/// count not divisible by `factor²`, or an output extent past `usize`.
pub fn plan_pixel_shuffle<R: Runtime<DType = DType>>(
    input: &Tensor<R>,
    factor: usize,
    layout: ConvLayout,
) -> Result<ImagePlan> {
    const OP: &str = "pixel_shuffle";
    let [n, c, h, w] = canonical_dims(input, layout, OP)?;
    check_factor(factor, OP)?;
    let r2 = factor_squared(factor, OP)?;
    if c % r2 != 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: format!(
                "{OP}: channel count {c} of input shape {:?} is not divisible by factor² = {r2}; \
                 pass a factor whose square divides the channel count",
                input.shape()
            ),
        });
    }
    let grow = |extent: usize, axis: &str| {
        extent.checked_mul(factor).ok_or_else(|| Error::InvalidArgument {
            arg: "factor",
            reason: format!(
                "{OP}: output {axis} = {extent} * {factor} overflows usize; pass a smaller factor"
            ),
        })
    };
    let out_dims = [n, c / r2, grow(h, "H")?, grow(w, "W")?];
    Ok(plan(
        OP,
        ImageMap::Shuffle { factor },
        layout,
        [n, c, h, w],
        out_dims,
    ))
}

/// Check `pixel_unshuffle` arguments and build its plan.
///
/// # Errors
///
/// `Error::InvalidArgument` for a rank other than 4, a zero factor, an H or W
/// not divisible by `factor`, or an output channel count past `usize`.
pub fn plan_pixel_unshuffle<R: Runtime<DType = DType>>(
    input: &Tensor<R>,
    factor: usize,
    layout: ConvLayout,
) -> Result<ImagePlan> {
    const OP: &str = "pixel_unshuffle";
    let [n, c, h, w] = canonical_dims(input, layout, OP)?;
    check_factor(factor, OP)?;
    let r2 = factor_squared(factor, OP)?;
    if h % factor != 0 || w % factor != 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: format!(
                "{OP}: spatial size H = {h}, W = {w} of input shape {:?} is not divisible by \
                 factor {factor}; crop or pad H and W to a multiple of the factor",
                input.shape()
            ),
        });
    }
    let out_c = c.checked_mul(r2).ok_or_else(|| Error::InvalidArgument {
        arg: "factor",
        reason: format!("{OP}: output C = {c} * {r2} overflows usize; pass a smaller factor"),
    })?;
    let out_dims = [n, out_c, h / factor, w / factor];
    Ok(plan(
        OP,
        ImageMap::Unshuffle { factor },
        layout,
        [n, c, h, w],
        out_dims,
    ))
}

/// Check the destination of an `_into` image op.
///
/// `out` must have the planned shape and the input's dtype, must not repeat
/// an element, and must not overlap `input`.
///
/// # Errors
///
/// `Error::DTypeMismatch`, `Error::ShapeMismatch`, or `Error::InvalidArgument`
/// for argument `out`.
pub fn validate_image_out<R: Runtime<DType = DType>>(
    out: &Tensor<R>,
    plan: &ImagePlan,
    input: &Tensor<R>,
) -> Result<()> {
    if out.dtype() != input.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: input.dtype(),
            rhs: out.dtype(),
        });
    }
    if out.shape() != plan.out_shape {
        return Err(Error::shape_mismatch(&plan.out_shape, out.shape()));
    }
    if out.numel() == 0 {
        return Ok(());
    }
    if has_broadcast_dim(out.shape(), out.strides()) {
        return Err(Error::InvalidArgument {
            arg: "out",
            reason: format!(
                "{}: out with strides {:?} repeats elements (stride 0 on a dimension > 1); \
                 pass a view that addresses each element once",
                plan.op,
                out.strides()
            ),
        });
    }
    check_out_disjoint_from(out, input, "input", plan.op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    fn zeros(shape: &[usize]) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::zeros(shape, DType::F32, &CpuDevice::new()).unwrap()
    }

    fn invalid_arg(result: Result<ImagePlan>) -> &'static str {
        match result {
            Err(Error::InvalidArgument { arg, .. }) => arg,
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn shuffle_plans_both_layouts() {
        let nchw = plan_pixel_shuffle(&zeros(&[2, 8, 3, 5]), 2, ConvLayout::Nchw).unwrap();
        assert_eq!(nchw.out_dims, [2, 2, 6, 10]);
        assert_eq!(nchw.out_shape, [2, 2, 6, 10]);
        let nhwc = plan_pixel_shuffle(&zeros(&[2, 3, 5, 8]), 2, ConvLayout::Nhwc).unwrap();
        assert_eq!(nhwc.in_dims, [2, 8, 3, 5]);
        assert_eq!(nhwc.out_shape, [2, 6, 10, 2]);
    }

    #[test]
    fn shuffle_rejects_bad_arguments() {
        assert_eq!(
            invalid_arg(plan_pixel_shuffle(&zeros(&[8, 3, 5]), 2, ConvLayout::Nchw)),
            "input"
        );
        assert_eq!(
            invalid_arg(plan_pixel_shuffle(
                &zeros(&[1, 8, 3, 5]),
                0,
                ConvLayout::Nchw
            )),
            "factor"
        );
        assert_eq!(
            invalid_arg(plan_pixel_shuffle(
                &zeros(&[1, 6, 3, 5]),
                2,
                ConvLayout::Nchw
            )),
            "input"
        );
    }

    #[test]
    fn unshuffle_plans_and_rejects_indivisible_extents() {
        let p = plan_pixel_unshuffle(&zeros(&[1, 4, 6, 10]), 2, ConvLayout::Nchw).unwrap();
        assert_eq!(p.out_dims, [1, 16, 3, 5]);
        assert_eq!(
            invalid_arg(plan_pixel_unshuffle(
                &zeros(&[1, 4, 6, 9]),
                2,
                ConvLayout::Nchw
            )),
            "input"
        );
        assert_eq!(
            invalid_arg(plan_pixel_unshuffle(
                &zeros(&[1, 4, 6, 6]),
                0,
                ConvLayout::Nchw
            )),
            "factor"
        );
    }

    #[test]
    fn out_must_match_and_stay_disjoint() {
        let x = zeros(&[1, 4, 2, 2]);
        let p = plan_pixel_shuffle(&x, 2, ConvLayout::Nchw).unwrap();
        assert!(validate_image_out(&zeros(&[1, 1, 4, 4]), &p, &x).is_ok());
        assert!(matches!(
            validate_image_out(&zeros(&[1, 1, 4, 3]), &p, &x),
            Err(Error::ShapeMismatch { .. })
        ));
        let y = x.reshape(&[1, 1, 4, 4]).unwrap();
        assert!(matches!(
            validate_image_out(&y, &p, &x),
            Err(Error::InvalidArgument { arg: "out", .. })
        ));
    }

    #[test]
    fn canonical_strides_reorder_nhwc() {
        assert_eq!(
            canonical_strides(&[60, 20, 4, 1], ConvLayout::Nhwc),
            [60, 1, 20, 4]
        );
        assert_eq!(
            canonical_strides(&[60, 15, 5, 1], ConvLayout::Nchw),
            [60, 15, 5, 1]
        );
    }
}
