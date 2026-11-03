//! Argument checks and pass schedule of `ImageOps::resize`.
//!
//! The schedule follows PyTorch's CPU kernels in
//! `aten/src/ATen/native/cpu/UpSampleKernel.cpp`:
//!
//! - Without antialias, `upsample_generic_Nd_kernel_impl` interpolates W
//!   inside H for every output, so a W pass into an intermediate in the
//!   accumulation type followed by an H pass gives the same values. Both
//!   passes run even when a size does not change.
//! - With antialias, `separable_upsample_generic_Nd_kernel_impl` runs the
//!   horizontal pass first, then the vertical pass, and skips an axis whose
//!   size does not change. Equal sizes on both axes copy the input.
//!
//! Every backend runs the same [`ResizeStep`]s with the same kernel mode.

use super::image_validation::{canonical_dims, layout_shape};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::traits::{ConvLayout, ResizeConfig, ResizeFilter};
use crate::runtime::Runtime;
use crate::runtime::common::view_span::{check_out_disjoint_from, has_broadcast_dim};
use crate::tensor::Tensor;

const OP: &str = "resize";

/// Window rule of one pass. Kernels receive [`ResizeMode::kernel_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    /// Copy the source element. The axis keeps its size.
    Copy,
    /// 2-tap linear interpolation, `antialias=False`.
    Linear,
    /// 4-tap cubic convolution with `a = -0.75`, `antialias=False`.
    Cubic,
    /// Triangle filter over the antialias window.
    AaLinear,
    /// Keys cubic filter with `a = -0.5` over the antialias window.
    AaCubic,
}

impl ResizeMode {
    /// Kernel mode word: 0 copy, 1 linear, 2 cubic, 3 AA linear, 4 AA cubic.
    #[cfg(any(feature = "cuda", feature = "wgpu"))]
    pub const fn kernel_mode(self) -> u32 {
        match self {
            Self::Copy => 0,
            Self::Linear => 1,
            Self::Cubic => 2,
            Self::AaLinear => 3,
            Self::AaCubic => 4,
        }
    }

    /// True for the antialias modes.
    pub const fn is_antialias(self) -> bool {
        matches!(self, Self::AaLinear | Self::AaCubic)
    }
}

/// One separable pass along H or W.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizeStep {
    /// Canonical axis: 2 for H, 3 for W.
    pub axis: usize,
    /// Window rule.
    pub mode: ResizeMode,
    /// Source extent along `axis`.
    pub in_size: usize,
    /// Destination extent along `axis`.
    pub out_size: usize,
}

/// Pass schedule of one resize call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeSchedule {
    /// One pass from the input to the output.
    One(ResizeStep),
    /// A W pass into an intermediate of extents `[N, C, H_in, W_out]` in the
    /// accumulation type, then an H pass into the output.
    Two {
        /// Pass along W.
        w: ResizeStep,
        /// Pass along H.
        h: ResizeStep,
    },
}

/// Validated geometry of one resize call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizePlan {
    /// Layout of input and output.
    pub layout: ConvLayout,
    /// Input extents as `[N, C, H, W]`.
    pub in_dims: [usize; 4],
    /// Output extents as `[N, C, H, W]`.
    pub out_dims: [usize; 4],
    /// Output shape in `layout` order.
    pub out_shape: [usize; 4],
    /// Passes to run.
    pub schedule: ResizeSchedule,
}

impl ResizePlan {
    /// Output element count.
    pub fn numel(&self) -> usize {
        self.out_dims.iter().product()
    }

    /// True when kernels walk tensors with C fastest (NHWC order).
    pub fn channels_last(&self) -> bool {
        self.layout == ConvLayout::Nhwc
    }

    /// Canonical extents of the intermediate of a two-pass schedule.
    pub fn mid_dims(&self) -> [usize; 4] {
        let [n, c, h, _] = self.in_dims;
        [n, c, h, self.out_dims[3]]
    }

    /// Shape in `layout` order of the intermediate of a two-pass schedule.
    /// Only the GPU backends allocate the intermediate as a tensor.
    #[cfg(any(feature = "cuda", feature = "wgpu"))]
    pub fn mid_shape(&self) -> [usize; 4] {
        layout_shape(self.mid_dims(), self.layout)
    }

    /// Canonical strides of a contiguous tensor of canonical extents `dims`
    /// stored in `layout` order.
    pub fn dense_strides(&self, dims: [usize; 4]) -> [isize; 4] {
        let [_, c, h, w] = dims.map(|d| d as isize);
        match self.layout {
            ConvLayout::Nchw => [c * h * w, h * w, w, 1],
            ConvLayout::Nhwc => [h * w * c, 1, w * c, c],
        }
    }
}

fn step(axis: usize, mode: ResizeMode, in_size: usize, out_size: usize) -> ResizeStep {
    ResizeStep {
        axis,
        mode,
        in_size,
        out_size,
    }
}

/// Check `resize` arguments and build its plan.
///
/// # Errors
///
/// `Error::InvalidArgument` for a rank other than 4, a zero input H or W, a
/// zero output size, or an output or intermediate element count past
/// `isize::MAX`.
pub fn plan_resize<R: Runtime<DType = DType>>(
    input: &Tensor<R>,
    config: &ResizeConfig,
    layout: ConvLayout,
) -> Result<ResizePlan> {
    let [n, c, h, w] = canonical_dims(input, layout, OP)?;
    if h == 0 || w == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: format!(
                "{OP}: input shape {:?} has H = {h}, W = {w}; H and W must be at least 1",
                input.shape()
            ),
        });
    }
    let (oh, ow) = config.size;
    if oh == 0 || ow == 0 {
        return Err(Error::InvalidArgument {
            arg: "config",
            reason: format!(
                "{OP}: output size ({oh}, {ow}) has a zero extent; pass sizes of at least 1"
            ),
        });
    }
    let out_dims = [n, c, oh, ow];
    // The output and the `[N, C, H_in, W_out]` intermediate must both be
    // addressable.
    let count = |dims: [usize; 4]| {
        dims.iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .filter(|&total| total <= isize::MAX as usize)
    };
    if count(out_dims).is_none() || count([n, c, h, ow]).is_none() {
        return Err(Error::InvalidArgument {
            arg: "config",
            reason: format!(
                "{OP}: output extents {out_dims:?} for input extents {:?} overflow the \
                 addressable element count; pass a smaller output size",
                [n, c, h, w]
            ),
        });
    }

    let schedule = if config.antialias {
        let mode = match config.filter {
            ResizeFilter::Bilinear => ResizeMode::AaLinear,
            ResizeFilter::Bicubic => ResizeMode::AaCubic,
        };
        match (ow != w, oh != h) {
            (true, true) => ResizeSchedule::Two {
                w: step(3, mode, w, ow),
                h: step(2, mode, h, oh),
            },
            (true, false) => ResizeSchedule::One(step(3, mode, w, ow)),
            (false, true) => ResizeSchedule::One(step(2, mode, h, oh)),
            (false, false) => ResizeSchedule::One(step(3, ResizeMode::Copy, w, w)),
        }
    } else {
        let mode = match config.filter {
            ResizeFilter::Bilinear => ResizeMode::Linear,
            ResizeFilter::Bicubic => ResizeMode::Cubic,
        };
        ResizeSchedule::Two {
            w: step(3, mode, w, ow),
            h: step(2, mode, h, oh),
        }
    };

    Ok(ResizePlan {
        layout,
        in_dims: [n, c, h, w],
        out_dims,
        out_shape: layout_shape(out_dims, layout),
        schedule,
    })
}

/// Check the destination of `resize_into`.
///
/// `out` must have the planned shape and the input's dtype, must not repeat
/// an element, and must not overlap `input`.
///
/// # Errors
///
/// `Error::DTypeMismatch`, `Error::ShapeMismatch`, or `Error::InvalidArgument`
/// for argument `out`.
pub fn validate_resize_out<R: Runtime<DType = DType>>(
    out: &Tensor<R>,
    plan: &ResizePlan,
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
                "{OP}: out with strides {:?} repeats elements (stride 0 on a dimension > 1); \
                 pass a view that addresses each element once",
                out.strides()
            ),
        });
    }
    check_out_disjoint_from(out, input, "input", OP)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    fn zeros(shape: &[usize]) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::zeros(shape, DType::F32, &CpuDevice::new()).unwrap()
    }

    fn config(size: (usize, usize), filter: ResizeFilter, aa: bool) -> ResizeConfig {
        ResizeConfig::new(size, filter).with_antialias(aa)
    }

    fn invalid_arg(result: Result<ResizePlan>) -> &'static str {
        match result {
            Err(Error::InvalidArgument { arg, .. }) => arg,
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn plain_filters_always_run_both_passes() {
        let x = zeros(&[1, 2, 4, 6]);
        let p = plan_resize(
            &x,
            &config((4, 6), ResizeFilter::Bicubic, false),
            ConvLayout::Nchw,
        )
        .unwrap();
        assert_eq!(
            p.schedule,
            ResizeSchedule::Two {
                w: step(3, ResizeMode::Cubic, 6, 6),
                h: step(2, ResizeMode::Cubic, 4, 4),
            }
        );
    }

    #[test]
    fn antialias_skips_unchanged_axes() {
        let x = zeros(&[1, 2, 4, 6]);
        let aa = |size| {
            plan_resize(
                &x,
                &config(size, ResizeFilter::Bilinear, true),
                ConvLayout::Nchw,
            )
            .unwrap()
            .schedule
        };
        assert_eq!(
            aa((4, 3)),
            ResizeSchedule::One(step(3, ResizeMode::AaLinear, 6, 3))
        );
        assert_eq!(
            aa((2, 6)),
            ResizeSchedule::One(step(2, ResizeMode::AaLinear, 4, 2))
        );
        assert_eq!(
            aa((4, 6)),
            ResizeSchedule::One(step(3, ResizeMode::Copy, 6, 6))
        );
    }

    #[test]
    fn nhwc_plans_permute_the_output_shape() {
        let x = zeros(&[2, 5, 7, 3]);
        let p = plan_resize(
            &x,
            &config((4, 9), ResizeFilter::Bilinear, false),
            ConvLayout::Nhwc,
        )
        .unwrap();
        assert_eq!(p.in_dims, [2, 3, 5, 7]);
        assert_eq!(p.out_shape, [2, 4, 9, 3]);
        assert_eq!(p.mid_dims(), [2, 3, 5, 9]);
        assert_eq!(p.dense_strides(p.mid_dims()), [135, 1, 27, 3]);
    }

    #[test]
    fn bad_arguments_name_their_argument() {
        let cfg = config((2, 2), ResizeFilter::Bicubic, false);
        assert_eq!(
            invalid_arg(plan_resize(&zeros(&[2, 2, 2]), &cfg, ConvLayout::Nchw)),
            "input"
        );
        assert_eq!(
            invalid_arg(plan_resize(&zeros(&[1, 1, 0, 2]), &cfg, ConvLayout::Nchw)),
            "input"
        );
        let zero = config((0, 2), ResizeFilter::Bicubic, false);
        assert_eq!(
            invalid_arg(plan_resize(&zeros(&[1, 1, 2, 2]), &zero, ConvLayout::Nchw)),
            "config"
        );
        let huge = config((usize::MAX, 2), ResizeFilter::Bicubic, false);
        assert_eq!(
            invalid_arg(plan_resize(&zeros(&[1, 1, 2, 2]), &huge, ConvLayout::Nchw)),
            "config"
        );
    }

    #[test]
    fn out_must_match_and_stay_disjoint() {
        let x = zeros(&[1, 1, 4, 4]);
        let cfg = config((2, 8), ResizeFilter::Bilinear, false);
        let p = plan_resize(&x, &cfg, ConvLayout::Nchw).unwrap();
        assert!(validate_resize_out(&zeros(&[1, 1, 2, 8]), &p, &x).is_ok());
        assert!(matches!(
            validate_resize_out(&zeros(&[1, 1, 8, 2]), &p, &x),
            Err(Error::ShapeMismatch { .. })
        ));
        let alias = x.reshape(&[1, 1, 2, 8]).unwrap();
        assert!(matches!(
            validate_resize_out(&alias, &p, &x),
            Err(Error::InvalidArgument { arg: "out", .. })
        ));
    }
}
