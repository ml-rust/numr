//! im2col2d kernel launcher.
//!
//! Gathers the conv2d receptive fields into the `[N, C_in*Kh*Kw, spatial_len]`
//! column buffer the GEMM formulation of conv2d contracts over. The buffer
//! holds a window of the flattened `H_out*W_out` output positions, so a large
//! output runs as several column chunks.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::launch_dims::launch_config;
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::{kernel_name, kernel_names};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;
use crate::runtime::cuda::kernels::conv::conv_arg_u32;

/// CUDA caps the y and z grid dimensions at 65535 blocks. Both axes carry a
/// grid-stride loop in the kernel, so the extents are clamped rather than
/// rejected.
const CUDA_MAX_GRID_YZ: usize = 65535;

/// Widest im2col2d block along the flattened spatial axis.
const IM2COL2D_BLOCK_MAX: u32 = 256;

/// `value` as a 32-bit im2col2d kernel argument, or an error naming the argument.
fn to_u32(value: usize, what: &'static str) -> Result<u32> {
    conv_arg_u32(value, what, "im2col2d")
}

/// Whether this dtype has an `im2col2d_*` kernel.
///
/// The float widths conv2d accepts, minus FP8: FP8 matmul accumulates in F32
/// and has its own conv2d kernel, so it stays on the direct path.
#[inline]
pub fn im2col2d_has_kernel(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16)
}

/// Launch the conv2d im2col kernel.
///
/// # Arguments
///
/// * `input_ptr` - Input tensor `(N, C_in, H, W)`
/// * `col_ptr` - Column buffer `(N, C_in*Kh*Kw, spatial_len)`
/// * `pad_top`, `pad_left` - Resolved TOP/LEFT padding
/// * `spatial_start`, `spatial_len` - Window of the flattened
///   `output_h * output_w` positions the buffer holds
///
/// # Errors
///
/// Returns `Error::InvalidArgument` when the window runs past
/// `output_h * output_w`, or when an extent does not fit the kernel's 32-bit
/// arguments.
///
/// # Safety
///
/// Both pointers must be valid device allocations of the sizes implied by the
/// shape arguments.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_im2col2d(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    col_ptr: u64,
    batch: usize,
    c_in: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    output_h: usize,
    output_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    dilation_h: usize,
    dilation_w: usize,
    spatial_start: usize,
    spatial_len: usize,
) -> Result<()> {
    let rows = c_in
        .checked_mul(kernel_h)
        .and_then(|r| r.checked_mul(kernel_w))
        .ok_or_else(|| Error::InvalidArgument {
            arg: "c_in * kernel_h * kernel_w",
            reason: format!(
                "im2col2d contraction {c_in} * {kernel_h} * {kernel_w} overflows usize; \
                 use a smaller kernel or fewer input channels"
            ),
        })?;
    let spatial = output_h
        .checked_mul(output_w)
        .ok_or_else(|| Error::InvalidArgument {
            arg: "output_h * output_w",
            reason: format!(
                "im2col2d output {output_h} x {output_w} overflows usize; split the input"
            ),
        })?;
    if batch == 0 || rows == 0 || spatial_len == 0 {
        return Ok(());
    }

    match spatial_start.checked_add(spatial_len) {
        Some(end) if end <= spatial => {}
        _ => {
            return Err(Error::InvalidArgument {
                arg: "spatial_len",
                reason: format!(
                    "im2col2d window [{spatial_start}, {spatial_start} + {spatial_len}) runs past \
                     the {output_h}x{output_w} output; pass a window inside 0..{spatial}"
                ),
            });
        }
    }

    if !im2col2d_has_kernel(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "im2col2d",
        });
    }

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::IM2COL2D_MODULE)?;
        let func = get_kernel_function(&module, &kernel_name("im2col2d", dtype))?;

        // Threads walk consecutive flattened output positions, so a short
        // spatial extent gets a narrow block instead of leaving most lanes idle.
        let spatial_len_u32 = to_u32(spatial_len, "spatial_len")?;
        let block_x = spatial_len_u32
            .min(IM2COL2D_BLOCK_MAX)
            .next_multiple_of(32)
            .max(32);
        let grid = (
            spatial_len_u32.div_ceil(block_x),
            rows.min(CUDA_MAX_GRID_YZ) as u32,
            batch.min(CUDA_MAX_GRID_YZ) as u32,
        );
        let cfg = launch_config(grid, (block_x, 1, 1), 0);

        // The kernel decodes rows and positions in 32 bits. `rows` and
        // `spatial` bound every index it forms in that width.
        to_u32(rows, "c_in * kernel_h * kernel_w")?;
        to_u32(spatial, "output_h * output_w")?;
        let batch_u32 = to_u32(batch, "batch")?;
        let c_in_u32 = to_u32(c_in, "c_in")?;
        let height_u32 = to_u32(height, "height")?;
        let width_u32 = to_u32(width, "width")?;
        let kernel_h_u32 = to_u32(kernel_h, "kernel_h")?;
        let kernel_w_u32 = to_u32(kernel_w, "kernel_w")?;
        let output_w_u32 = to_u32(output_w, "output_w")?;
        let stride_h_u32 = to_u32(stride_h, "stride_h")?;
        let stride_w_u32 = to_u32(stride_w, "stride_w")?;
        let pad_top_u32 = to_u32(pad_top, "pad_top")?;
        let pad_left_u32 = to_u32(pad_left, "pad_left")?;
        let dilation_h_u32 = to_u32(dilation_h, "dilation_h")?;
        let dilation_w_u32 = to_u32(dilation_w, "dilation_w")?;
        let spatial_start_u32 = to_u32(spatial_start, "spatial_start")?;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&col_ptr);
        builder.arg(&batch_u32);
        builder.arg(&c_in_u32);
        builder.arg(&height_u32);
        builder.arg(&width_u32);
        builder.arg(&kernel_h_u32);
        builder.arg(&kernel_w_u32);
        builder.arg(&output_w_u32);
        builder.arg(&stride_h_u32);
        builder.arg(&stride_w_u32);
        builder.arg(&pad_top_u32);
        builder.arg(&pad_left_u32);
        builder.arg(&dilation_h_u32);
        builder.arg(&dilation_w_u32);
        builder.arg(&spatial_start_u32);
        builder.arg(&spatial_len_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA im2col2d kernel launch failed: {:?}", e)))?;
    }

    Ok(())
}
