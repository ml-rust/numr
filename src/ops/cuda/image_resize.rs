//! CUDA implementation of `ImageOps::resize`.
//!
//! Strided input and output views pass to the kernels with their strides. A
//! two-pass schedule writes a dense device intermediate in the accumulation
//! type: F32 for F32, F16 and BF16, F64 for F64. No data leaves the device.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::common::{
    ResizePlan, ResizeSchedule, ResizeStep, canonical_strides, plan_resize, validate_resize_out,
};
use crate::ops::{ConvLayout, ResizeConfig};
use crate::runtime::cuda::kernels::{ResizeLaunch, launch_resize_pass};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Accumulation dtype of `dtype`, or `UnsupportedDType`.
fn acc_dtype(dtype: DType) -> Result<DType> {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => Ok(DType::F32),
        DType::F64 => Ok(DType::F64),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "resize",
        }),
    }
}

/// A device view: address of its first element, dtype and canonical strides.
#[derive(Clone, Copy)]
struct View {
    ptr: u64,
    dtype: DType,
    strides: [isize; 4],
}

impl CudaClient {
    fn resize_step(
        &self,
        plan: &ResizePlan,
        step: ResizeStep,
        src: View,
        dst: View,
        dst_dims: [usize; 4],
    ) -> Result<()> {
        let launch = ResizeLaunch {
            src_dtype: src.dtype,
            dst_dtype: dst.dtype,
            dst_dims,
            src_strides: src.strides,
            dst_strides: dst.strides,
            channels_last: plan.channels_last(),
            axis: step.axis,
            mode: step.mode.kernel_mode(),
            in_size: step.in_size,
            out_size: step.out_size,
        };
        // SAFETY: each view's address folds in its offset, its strides match
        // its extents, and the destination was checked against repeats and
        // overlap with the source, or is a fresh intermediate.
        unsafe {
            launch_resize_pass(
                &self.context,
                &self.stream,
                self.device.index,
                src.ptr,
                dst.ptr,
                &launch,
            )
        }
    }

    /// Run `plan` from `input` into `out`, which already passed the checks.
    fn resize_run(
        &self,
        plan: &ResizePlan,
        input: &Tensor<CudaRuntime>,
        out: &Tensor<CudaRuntime>,
    ) -> Result<()> {
        let acc = acc_dtype(input.dtype())?;
        if plan.numel() == 0 {
            return Ok(());
        }
        let src = View {
            ptr: input.ptr(),
            dtype: input.dtype(),
            strides: canonical_strides(input.strides(), plan.layout),
        };
        let dst = View {
            ptr: out.ptr(),
            dtype: out.dtype(),
            strides: canonical_strides(out.strides(), plan.layout),
        };
        match plan.schedule {
            ResizeSchedule::One(step) => self.resize_step(plan, step, src, dst, plan.out_dims),
            ResizeSchedule::Two { w, h } => {
                let mid_dims = plan.mid_dims();
                let mid = Tensor::<CudaRuntime>::empty(&plan.mid_shape(), acc, &self.device)?;
                let mid_view = View {
                    ptr: mid.ptr(),
                    dtype: acc,
                    strides: plan.dense_strides(mid_dims),
                };
                self.resize_step(plan, w, src, mid_view, mid_dims)?;
                self.resize_step(plan, h, mid_view, dst, plan.out_dims)
            }
        }
    }

    /// Resize `input` into a new contiguous tensor.
    pub(crate) fn resize_alloc(
        &self,
        input: &Tensor<CudaRuntime>,
        config: &ResizeConfig,
        layout: ConvLayout,
    ) -> Result<Tensor<CudaRuntime>> {
        let plan = plan_resize(input, config, layout)?;
        acc_dtype(input.dtype())?;
        let out = Tensor::<CudaRuntime>::empty(&plan.out_shape, input.dtype(), &self.device)?;
        self.resize_run(&plan, input, &out)?;
        Ok(out)
    }

    /// Resize `input` into the caller's view `out`.
    pub(crate) fn resize_into_view(
        &self,
        out: &Tensor<CudaRuntime>,
        input: &Tensor<CudaRuntime>,
        config: &ResizeConfig,
        layout: ConvLayout,
    ) -> Result<()> {
        let plan = plan_resize(input, config, layout)?;
        acc_dtype(input.dtype())?;
        validate_resize_out(out, &plan, input)?;
        self.resize_run(&plan, input, out)
    }
}
