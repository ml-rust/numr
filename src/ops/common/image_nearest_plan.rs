//! Argument checks and plan of `upsample_nearest2d`.

use super::image_nearest::NearestAxis;
use super::image_validation::{ImageMap, ImagePlan, canonical_dims, plan};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::traits::{ConvLayout, NearestSize};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Output extent of one axis for scale factor `s`: `floor(in * s)` in f64.
fn scaled_extent(in_size: usize, s: f64, axis: &str) -> Result<usize> {
    const OP: &str = "upsample_nearest2d";
    if !s.is_finite() || s <= 0.0 {
        return Err(Error::InvalidArgument {
            arg: "size",
            reason: format!(
                "{OP}: scale factor {s} for {axis} is not finite and positive; pass a factor > 0"
            ),
        });
    }
    let out = (in_size as f64 * s).floor();
    if out < 1.0 || out > isize::MAX as f64 {
        return Err(Error::InvalidArgument {
            arg: "size",
            reason: format!(
                "{OP}: scale factor {s} on input {axis} = {in_size} gives output {axis} = {out}; \
                 pass a factor that gives an output size from 1 to isize::MAX"
            ),
        });
    }
    Ok(out as usize)
}

/// Check `upsample_nearest2d` arguments and build its plan.
///
/// # Errors
///
/// `Error::InvalidArgument` for a rank other than 4, a zero H or W, a scale
/// factor that is not finite and positive, or an output size of 0.
pub fn plan_upsample_nearest2d<R: Runtime<DType = DType>>(
    input: &Tensor<R>,
    size: NearestSize,
    layout: ConvLayout,
) -> Result<ImagePlan> {
    const OP: &str = "upsample_nearest2d";
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
    let (out_h, out_w, axis_h, axis_w) = match size {
        NearestSize::Scale(sh, sw) => {
            let out_h = scaled_extent(h, sh, "H")?;
            let out_w = scaled_extent(w, sw, "W")?;
            (
                out_h,
                out_w,
                NearestAxis::new(h, out_h, Some(sh)),
                NearestAxis::new(w, out_w, Some(sw)),
            )
        }
        NearestSize::Size(out_h, out_w) => {
            if out_h == 0 || out_w == 0 {
                return Err(Error::InvalidArgument {
                    arg: "size",
                    reason: format!(
                        "{OP}: output size ({out_h}, {out_w}) has a zero extent; pass sizes of at least 1"
                    ),
                });
            }
            (
                out_h,
                out_w,
                NearestAxis::new(h, out_h, None),
                NearestAxis::new(w, out_w, None),
            )
        }
    };
    let map = ImageMap::Nearest {
        h: axis_h,
        w: axis_w,
    };
    Ok(plan(OP, map, layout, [n, c, h, w], [n, c, out_h, out_w]))
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
    fn nearest_plans_scale_and_size() {
        let x = zeros(&[1, 2, 3, 5]);
        let s =
            plan_upsample_nearest2d(&x, NearestSize::Scale(2.0, 1.7), ConvLayout::Nchw).unwrap();
        assert_eq!(s.out_dims, [1, 2, 6, 8]);
        let z = plan_upsample_nearest2d(&x, NearestSize::Size(7, 3), ConvLayout::Nchw).unwrap();
        assert_eq!(z.out_dims, [1, 2, 7, 3]);
    }

    #[test]
    fn nearest_rejects_bad_sizes() {
        let x = zeros(&[1, 2, 3, 5]);
        let bad = [
            NearestSize::Scale(0.0, 1.0),
            NearestSize::Scale(f64::NAN, 1.0),
            NearestSize::Scale(1.0, 0.1),
            NearestSize::Size(0, 4),
        ];
        for size in bad {
            assert_eq!(
                invalid_arg(plan_upsample_nearest2d(&x, size, ConvLayout::Nchw)),
                "size"
            );
        }
        let empty = zeros(&[1, 2, 0, 5]);
        assert_eq!(
            invalid_arg(plan_upsample_nearest2d(
                &empty,
                NearestSize::Size(2, 2),
                ConvLayout::Nchw
            )),
            "input"
        );
    }
}
