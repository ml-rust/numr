//! CPU dispatch of `ImageOps::resize`.
//!
//! Strided input and output views pass straight to the pass kernel with their
//! strides. A two-pass schedule writes a dense intermediate in the
//! accumulation type: f32 for F32, F16 and BF16, f64 for F64.

use super::super::kernels::{AxisTaps, ResizeAcc, ResizePassGeometry, resize_pass_kernel};
use super::super::{CpuClient, CpuRuntime};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::ops::common::{
    ResizePlan, ResizeSchedule, ResizeStep, canonical_strides, plan_resize, validate_resize_out,
};
use crate::ops::{ConvLayout, ResizeConfig};
use crate::tensor::Tensor;

const OP: &str = "resize";

/// One pass from `src` to `dst` along `step.axis`.
///
/// # Safety
///
/// `src` and `dst` point at the first element of views whose canonical
/// strides are `src_strides` and `dst_strides` and whose destination
/// extents are `dst_dims`. The views share no element.
unsafe fn run_step<I: Element, O: Element, A: ResizeAcc>(
    client: &CpuClient,
    plan: &ResizePlan,
    step: ResizeStep,
    (src, src_strides): (*const I, [isize; 4]),
    (dst, dst_strides): (*mut O, [isize; 4]),
    dst_dims: [usize; 4],
) {
    let taps = AxisTaps::<A>::build(step.mode, step.in_size, step.out_size);
    let geometry = ResizePassGeometry {
        dst_dims,
        src_strides,
        dst_strides,
        axis: step.axis,
        channels_last: plan.channels_last(),
    };
    // SAFETY: forwarded from this function's contract.
    unsafe { resize_pass_kernel(client, src, dst, &geometry, step.mode, &taps) }
}

/// Run `plan` with elements `T` and accumulation type `A`.
fn run_typed<T: Element, A: ResizeAcc>(
    client: &CpuClient,
    plan: &ResizePlan,
    input: &Tensor<CpuRuntime>,
    out: &Tensor<CpuRuntime>,
) {
    let src = (
        input.ptr() as *const T,
        canonical_strides(input.strides(), plan.layout),
    );
    let dst = (
        out.ptr() as *mut T,
        canonical_strides(out.strides(), plan.layout),
    );
    match plan.schedule {
        // SAFETY: `ptr()` folds in each view's offset, the plan matches both
        // shapes, and `out` was checked against repeats and overlap.
        ResizeSchedule::One(step) => unsafe {
            run_step::<T, T, A>(client, plan, step, src, dst, plan.out_dims)
        },
        ResizeSchedule::Two { w, h } => {
            let mid_dims = plan.mid_dims();
            let mid_strides = plan.dense_strides(mid_dims);
            let mut mid = vec![A::constant(0.0); mid_dims.iter().product()];
            // SAFETY: as above, and `mid` is a dense buffer of `mid_dims`
            // that no tensor shares.
            unsafe {
                run_step::<T, A, A>(
                    client,
                    plan,
                    w,
                    src,
                    (mid.as_mut_ptr(), mid_strides),
                    mid_dims,
                );
                run_step::<A, T, A>(
                    client,
                    plan,
                    h,
                    (mid.as_ptr(), mid_strides),
                    dst,
                    plan.out_dims,
                );
            }
        }
    }
}

/// Run `plan` from `input` into `out`, which already passed the checks.
fn run(
    client: &CpuClient,
    plan: &ResizePlan,
    input: &Tensor<CpuRuntime>,
    out: &Tensor<CpuRuntime>,
) -> Result<()> {
    if plan.numel() == 0 {
        return Ok(());
    }
    match input.dtype() {
        DType::F32 => run_typed::<f32, f32>(client, plan, input, out),
        DType::F64 => run_typed::<f64, f64>(client, plan, input, out),
        #[cfg(feature = "f16")]
        DType::F16 => run_typed::<half::f16, f32>(client, plan, input, out),
        #[cfg(feature = "f16")]
        DType::BF16 => run_typed::<half::bf16, f32>(client, plan, input, out),
        dtype => return Err(Error::UnsupportedDType { dtype, op: OP }),
    }
    Ok(())
}

fn check_dtype(dtype: DType) -> Result<()> {
    let supported = matches!(dtype, DType::F32 | DType::F64)
        || (cfg!(feature = "f16") && matches!(dtype, DType::F16 | DType::BF16));
    if supported {
        Ok(())
    } else {
        Err(Error::UnsupportedDType { dtype, op: OP })
    }
}

/// Resize `input` into a new contiguous tensor.
pub fn resize_alloc(
    client: &CpuClient,
    input: &Tensor<CpuRuntime>,
    config: &ResizeConfig,
    layout: ConvLayout,
) -> Result<Tensor<CpuRuntime>> {
    let plan = plan_resize(input, config, layout)?;
    check_dtype(input.dtype())?;
    let out = Tensor::<CpuRuntime>::empty(&plan.out_shape, input.dtype(), &client.device)?;
    run(client, &plan, input, &out)?;
    Ok(out)
}

/// Resize `input` into the caller's view `out`.
pub fn resize_into(
    client: &CpuClient,
    out: &Tensor<CpuRuntime>,
    input: &Tensor<CpuRuntime>,
    config: &ResizeConfig,
    layout: ConvLayout,
) -> Result<()> {
    let plan = plan_resize(input, config, layout)?;
    check_dtype(input.dtype())?;
    validate_resize_out(out, &plan, input)?;
    run(client, &plan, input, out)
}
