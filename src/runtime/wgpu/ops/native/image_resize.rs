//! WebGPU dispatch of `ImageOps::resize`. F32 only.
//!
//! Strided input and output views pass to the kernel as a base word and
//! strides. A two-pass schedule writes a dense F32 intermediate on the device.

use super::helpers::*;
use super::image::{stride_words, word};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::common::{
    ResizePlan, ResizeSchedule, ResizeStep, canonical_strides, plan_resize, validate_resize_out,
};
use crate::ops::{ConvLayout, ResizeConfig};
use crate::runtime::wgpu::client::get_buffer;
use crate::runtime::wgpu::shaders::image_resize::launch_resize_pass;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

const OP: &str = "resize";

fn check_dtype(dtype: DType) -> Result<()> {
    if dtype == DType::F32 {
        Ok(())
    } else {
        Err(Error::UnsupportedDType { dtype, op: OP })
    }
}

/// One pass from `src` to `dst`, each a tensor with canonical strides.
fn run_step(
    client: &WgpuClient,
    plan: &ResizePlan,
    step: ResizeStep,
    (src, src_strides): (&Tensor<WgpuRuntime>, [isize; 4]),
    (dst, dst_strides): (&Tensor<WgpuRuntime>, [isize; 4]),
    dst_dims: [usize; 4],
) -> Result<()> {
    let total: usize = dst_dims.iter().product();
    if total == 0 {
        return Ok(());
    }
    let [_, dim_c, dim_h, dim_w] = dst_dims;
    let scale = step.in_size as f32 / step.out_size as f32;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let params = ResizeParams {
        total: word(total, "element count", OP)?,
        dim_c: word(dim_c, "output C", OP)?,
        dim_h: word(dim_h, "output H", OP)?,
        dim_w: word(dim_w, "output W", OP)?,
        channels_last: u32::from(plan.channels_last()),
        axis: step.axis as u32,
        mode: step.mode.kernel_mode(),
        in_size: word(step.in_size, "input extent", OP)?,
        out_size: word(step.out_size, "output extent", OP)?,
        src_base: word(src.offset(), "input offset", OP)?,
        dst_base: word(dst.offset(), "output offset", OP)?,
        scale,
        invscale,
        src_strides: stride_words(src_strides, OP)?,
        dst_strides: stride_words(dst_strides, OP)?,
        _pad: [0; 3],
    };
    let src_ptr = src.storage().ptr();
    let dst_ptr = dst.storage().ptr();
    let src_buf = get_buffer(src_ptr).ok_or_else(|| {
        Error::Internal(format!(
            "{OP}: input buffer {src_ptr} is not in the wgpu buffer registry"
        ))
    })?;
    let dst_buf = get_buffer(dst_ptr).ok_or_else(|| {
        Error::Internal(format!(
            "{OP}: output buffer {dst_ptr} is not in the wgpu buffer registry"
        ))
    })?;
    let params_buf = create_params_buffer(client, &params);
    launch_resize_pass(
        client,
        &src_buf,
        &dst_buf,
        dst.shares_storage_with(src),
        &params_buf,
        total,
    )
}

/// Run `plan` from `input` into `out`, which already passed the checks.
fn run(
    client: &WgpuClient,
    plan: &ResizePlan,
    input: &Tensor<WgpuRuntime>,
    out: &Tensor<WgpuRuntime>,
) -> Result<()> {
    if plan.numel() == 0 {
        return Ok(());
    }
    let src = (input, canonical_strides(input.strides(), plan.layout));
    let dst = (out, canonical_strides(out.strides(), plan.layout));
    match plan.schedule {
        ResizeSchedule::One(step) => run_step(client, plan, step, src, dst, plan.out_dims),
        ResizeSchedule::Two { w, h } => {
            let mid_dims = plan.mid_dims();
            let mid = alloc_output(client, &plan.mid_shape(), DType::F32)?;
            let mid_view = (&mid, plan.dense_strides(mid_dims));
            run_step(client, plan, w, src, mid_view, mid_dims)?;
            run_step(client, plan, h, mid_view, dst, plan.out_dims)
        }
    }
}

/// Resize `input` into a new contiguous tensor.
pub(crate) fn native_resize(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    config: &ResizeConfig,
    layout: ConvLayout,
) -> Result<Tensor<WgpuRuntime>> {
    let plan = plan_resize(input, config, layout)?;
    check_dtype(input.dtype())?;
    let out = alloc_output(client, &plan.out_shape, input.dtype())?;
    run(client, &plan, input, &out)?;
    Ok(out)
}

/// Resize `input` into the caller's view `out`.
pub(crate) fn native_resize_into(
    client: &WgpuClient,
    out: &Tensor<WgpuRuntime>,
    input: &Tensor<WgpuRuntime>,
    config: &ResizeConfig,
    layout: ConvLayout,
) -> Result<()> {
    let plan = plan_resize(input, config, layout)?;
    check_dtype(input.dtype())?;
    validate_resize_out(out, &plan, input)?;
    run(client, &plan, input, out)
}
