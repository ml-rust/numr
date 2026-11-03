//! WebGPU dispatch for the `ImageOps` gather ops. F32, I32 and U32.
//!
//! Strided input and output views pass to the kernel as a base word and
//! strides, so no operand is copied to contiguous first.

use super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::common::{ImageMap, ImagePlan, canonical_strides, validate_image_out};
use crate::runtime::wgpu::client::get_buffer;
use crate::runtime::wgpu::shaders::image::{ImageEntry, launch_image};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Narrow `value` to u32, or name `what` in a `BackendLimitation`.
pub(super) fn word(value: usize, what: &str, op: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::backend_limitation(
            "WebGPU",
            op,
            format!("{what} {value} exceeds the 32-bit addressing of the WebGPU kernel"),
        )
    })
}

/// Pack strides as two's complement words.
pub(super) fn stride_words(strides: [isize; 4], op: &'static str) -> Result<[u32; 4]> {
    let mut words = [0u32; 4];
    for (slot, stride) in words.iter_mut().zip(strides) {
        let stride = i32::try_from(stride).map_err(|_| {
            Error::backend_limitation(
                "WebGPU",
                op,
                format!("stride {stride} does not fit the 32-bit addressing of the WebGPU kernel"),
            )
        })?;
        // The kernel adds strides as wrapping u32, which equals i32 addition.
        *slot = stride as u32;
    }
    Ok(words)
}

fn check_dtype(dtype: DType, op: &'static str) -> Result<()> {
    if matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
        Ok(())
    } else {
        Err(Error::UnsupportedDType { dtype, op })
    }
}

/// Run `plan` from `input` into `out`.
///
/// `out` has the planned shape and the input's dtype, addresses each element
/// once, and shares none with `input`.
fn run(
    client: &WgpuClient,
    plan: &ImagePlan,
    input: &Tensor<WgpuRuntime>,
    out: &Tensor<WgpuRuntime>,
) -> Result<()> {
    let op = plan.op;
    let total = plan.numel();
    if total == 0 {
        return Ok(());
    }
    let (entry, factor, nearest) = match plan.map {
        ImageMap::Shuffle { factor } => (ImageEntry::PixelShuffle, factor, [(0, 0.0); 2]),
        ImageMap::Unshuffle { factor } => (ImageEntry::PixelUnshuffle, factor, [(0, 0.0); 2]),
        ImageMap::Nearest { h, w } => (
            ImageEntry::UpsampleNearest2d,
            1,
            [h.kernel_args(), w.kernel_args()],
        ),
    };
    let [_, dim_c, dim_h, dim_w] = plan.out_dims;
    let params = ImageParams {
        total: word(total, "element count", op)?,
        dim_c: word(dim_c, "output C", op)?,
        dim_h: word(dim_h, "output H", op)?,
        dim_w: word(dim_w, "output W", op)?,
        channels_last: u32::from(plan.channels_last()),
        factor: word(factor, "factor", op)?,
        in_h: word(plan.in_dims[2], "input H", op)?,
        in_w: word(plan.in_dims[3], "input W", op)?,
        mode_h: nearest[0].0,
        mode_w: nearest[1].0,
        scale_h: nearest[0].1,
        scale_w: nearest[1].1,
        src_base: word(input.offset(), "input offset", op)?,
        dst_base: word(out.offset(), "output offset", op)?,
        src_strides: stride_words(canonical_strides(input.strides(), plan.layout), op)?,
        dst_strides: stride_words(canonical_strides(out.strides(), plan.layout), op)?,
        _pad: [0; 2],
    };
    let src_ptr = input.storage().ptr();
    let dst_ptr = out.storage().ptr();
    let src_buf = get_buffer(src_ptr).ok_or_else(|| {
        Error::Internal(format!(
            "{op}: input buffer {src_ptr} is not in the wgpu buffer registry"
        ))
    })?;
    let dst_buf = get_buffer(dst_ptr).ok_or_else(|| {
        Error::Internal(format!(
            "{op}: output buffer {dst_ptr} is not in the wgpu buffer registry"
        ))
    })?;
    let params_buf = create_params_buffer(client, &params);

    launch_image(
        client,
        entry,
        input.dtype(),
        &src_buf,
        &dst_buf,
        out.shares_storage_with(input),
        &params_buf,
        total,
    )
}

/// Allocate a contiguous output for `plan` and fill it.
pub(crate) fn native_image(
    client: &WgpuClient,
    plan: &ImagePlan,
    input: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    check_dtype(input.dtype(), plan.op)?;
    let out = alloc_output(client, &plan.out_shape, input.dtype())?;
    run(client, plan, input, &out)?;
    Ok(out)
}

/// Check `out` against `plan` and fill it.
pub(crate) fn native_image_into(
    client: &WgpuClient,
    plan: &ImagePlan,
    out: &Tensor<WgpuRuntime>,
    input: &Tensor<WgpuRuntime>,
) -> Result<()> {
    check_dtype(input.dtype(), plan.op)?;
    validate_image_out(out, plan, input)?;
    run(client, plan, input, out)
}
