//! conv_transpose1d GEMM-first fold launcher.
//!
//! Sums, for every output position, the taps of the `[N, L, C_out*K]` GEMM
//! product that land on it, adds the bias, and writes `[N, C_out, L_out]`.
//! Every output element is written once by one thread, so no atomics are
//! involved. The gather-first sibling is `col_transpose1d`.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::col_transpose1d::col_transpose1d_has_kernel;
use super::launch_dims::launch_config;
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::{kernel_name, kernel_names};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// CUDA caps the y and z grid dimensions at 65535 blocks. Both axes carry a
/// grid-stride loop in the kernel, so the extents are clamped rather than
/// rejected.
const CUDA_MAX_GRID_YZ: usize = 65535;

/// Widest block along the output axis.
const COL2IM_TRANSPOSE1D_BLOCK_MAX: u32 = 256;

/// Launch the conv_transpose1d GEMM-first fold kernel.
///
/// # Arguments
///
/// * `col_ptr` - GEMM product `(N, L, C_out*K)`
/// * `bias_ptr` - Optional bias `(C_out)`
/// * `out_ptr` - Output tensor `(N, C_out, L_out)`
/// * `pad_left` - Resolved LEFT padding
///
/// # Safety
///
/// Every pointer must be a valid device allocation of the size implied by the
/// shape arguments.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_col2im_transpose1d(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    col_ptr: u64,
    bias_ptr: Option<u64>,
    out_ptr: u64,
    batch: usize,
    length: usize,
    c_out: usize,
    kernel_size: usize,
    output_length: usize,
    stride: usize,
    pad_left: usize,
    dilation: usize,
) -> Result<()> {
    if batch == 0 || c_out == 0 || output_length == 0 {
        return Ok(());
    }

    // A zero stride would divide by zero in the kernel; validation rules it
    // out, so this only guards against a caller that skipped validation.
    if stride == 0 {
        return Err(Error::Internal(
            "col2im_transpose1d requires stride >= 1".to_string(),
        ));
    }

    // Same dtype set as the gather kernel: the two are the two halves of one
    // formulation and ship together.
    if !col_transpose1d_has_kernel(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "col2im_transpose1d",
        });
    }

    unsafe {
        let module = get_or_load_module(
            context,
            device_index,
            kernel_names::COL2IM_TRANSPOSE1D_MODULE,
        )?;
        let func = get_kernel_function(&module, &kernel_name("col2im_transpose1d", dtype))?;

        // Threads walk consecutive output positions, so a short row gets a
        // narrow block instead of leaving most lanes idle.
        let block_x = (output_length as u32)
            .next_multiple_of(32)
            .clamp(32, COL2IM_TRANSPOSE1D_BLOCK_MAX);
        let grid = (
            (output_length as u32).div_ceil(block_x),
            c_out.min(CUDA_MAX_GRID_YZ) as u32,
            batch.min(CUDA_MAX_GRID_YZ) as u32,
        );
        let cfg = launch_config(grid, (block_x, 1, 1), 0);

        // A null bias pointer tells the kernel there is no bias.
        let bias_raw: u64 = bias_ptr.unwrap_or(0);
        let batch_u32 = batch as u32;
        let length_u32 = length as u32;
        let c_out_u32 = c_out as u32;
        let kernel_size_u32 = kernel_size as u32;
        let output_length_u32 = output_length as u32;
        let stride_u32 = stride as u32;
        let pad_left_u32 = pad_left as u32;
        let dilation_u32 = dilation as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&col_ptr);
        builder.arg(&bias_raw);
        builder.arg(&out_ptr);
        builder.arg(&batch_u32);
        builder.arg(&length_u32);
        builder.arg(&c_out_u32);
        builder.arg(&kernel_size_u32);
        builder.arg(&output_length_u32);
        builder.arg(&stride_u32);
        builder.arg(&pad_left_u32);
        builder.arg(&dilation_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA col2im_transpose1d kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}
