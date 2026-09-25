//! conv_transpose1d CUDA kernel launcher.
//!
//! One kernel, gather form: each thread owns one output element and sums its
//! taps in a fixed order, so the result never depends on the launch shape.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    launch_config,
};
use super::tuning::CONV_MODULE;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;

/// Launch conv_transpose1d kernel.
///
/// # Arguments
///
/// * `input_ptr` - Input tensor (N, C_in, L)
/// * `weight_ptr` - Weight tensor (C_in, C_out/groups, K) — input channels lead
/// * `bias_ptr` - Optional bias tensor (C_out,)
/// * `output_ptr` - Output tensor (N, C_out, L_out)
/// * `padding` - Resolved LEFT padding; for transposed conv this TRIMS the output
///
/// # Safety
///
/// Caller must ensure all pointers are valid device allocations of the sizes
/// implied by the shape arguments.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_conv_transpose1d(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    weight_ptr: u64,
    bias_ptr: Option<u64>,
    output_ptr: u64,
    batch: usize,
    c_in: usize,
    length: usize,
    c_out: usize,
    kernel_size: usize,
    output_length: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<()> {
    let total = batch * c_out * output_length;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, CONV_MODULE)?;
        let func_name = kernel_name("conv_transpose1d", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let batch_u32 = batch as u32;
        let c_in_u32 = c_in as u32;
        let length_u32 = length as u32;
        let c_out_u32 = c_out as u32;
        let kernel_size_u32 = kernel_size as u32;
        let output_length_u32 = output_length as u32;
        let stride_u32 = stride as u32;
        let padding_u32 = padding as u32;
        let dilation_u32 = dilation as u32;
        let groups_u32 = groups as u32;
        let has_bias_u32: u32 = if bias_ptr.is_some() { 1 } else { 0 };
        let bias_ptr_val = bias_ptr.unwrap_or(0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&bias_ptr_val);
        builder.arg(&output_ptr);
        builder.arg(&batch_u32);
        builder.arg(&c_in_u32);
        builder.arg(&length_u32);
        builder.arg(&c_out_u32);
        builder.arg(&kernel_size_u32);
        builder.arg(&output_length_u32);
        builder.arg(&stride_u32);
        builder.arg(&padding_u32);
        builder.arg(&dilation_u32);
        builder.arg(&groups_u32);
        builder.arg(&has_bias_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA conv_transpose1d kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}
