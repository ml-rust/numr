//! depthwise_conv2d CUDA kernel launcher.
//!
//! Two kernels serve one op: the flat `depthwise_conv2d` and the
//! column-blocked `depthwise_conv2d_ox`. [`depthwise_conv2d_variant`] picks
//! between them from the shape and the device. Both form one output element's
//! sum over `ky` ascending, then `kx` ascending, in the same accumulator
//! width, with bias added last, so the choice moves no bits.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    launch_config,
};
use super::tuning::{
    CONV_BLOCK_THREADS, CONV_MODULE, CUDA_MAX_GRID_YZ, DEPTHWISE_CONV2D_OX_BLOCK,
    DEPTHWISE_CONV2D_OX_MIN_OUTPUT_WIDTH, DEPTHWISE_CONV2D_OX_MIN_WAVES,
    DEPTHWISE_CONV2D_OX_MODULE, fills_device,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Device;
use crate::runtime::cuda::CudaDevice;
use crate::runtime::cuda::GuardedStream;

/// Which depthwise_conv2d kernel a launch runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DepthwiseConv2dVariant {
    /// `depthwise_conv2d`: one output element per thread.
    Flat,
    /// `depthwise_conv2d_ox`: [`DEPTHWISE_CONV2D_OX_BLOCK`] output columns per
    /// thread.
    Ox,
}

/// The kernel this shape runs on.
///
/// Register-block over output columns when the row is wide enough and the
/// blocked grid still fills the device. FP8 keeps its own legacy flat kernel,
/// as it does for conv1d.
fn depthwise_conv2d_variant(
    device_index: usize,
    dtype: DType,
    batch: usize,
    channels: usize,
    output_h: usize,
    output_w: usize,
) -> DepthwiseConv2dVariant {
    if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
        return DepthwiseConv2dVariant::Flat;
    }

    let compute_units = CudaDevice::new(device_index).profile().compute_units as usize;
    let x_extent = output_w.div_ceil(DEPTHWISE_CONV2D_OX_BLOCK);
    // The blocked kernel folds (oy, column-block) onto the x axis and keeps
    // channel and batch on y and z.
    let x_work = output_h.saturating_mul(x_extent);
    let ox_threads = batch.saturating_mul(channels).saturating_mul(x_work);

    let blocked = output_w >= DEPTHWISE_CONV2D_OX_MIN_OUTPUT_WIDTH
        && fills_device(ox_threads, compute_units, DEPTHWISE_CONV2D_OX_MIN_WAVES)
        // A shape that overflows the channel y axis or the batch z axis falls
        // back to the flat kernel instead of failing. The x axis carries the
        // folded work and is bounded far higher, so it needs no gate.
        && channels <= CUDA_MAX_GRID_YZ
        && batch <= CUDA_MAX_GRID_YZ;

    if blocked {
        DepthwiseConv2dVariant::Ox
    } else {
        DepthwiseConv2dVariant::Flat
    }
}

/// Launch depthwise_conv2d kernel.
///
/// Performs depthwise 2D convolution where each channel is convolved independently.
///
/// # Arguments
///
/// * `input_ptr` - Input tensor (N, C, H, W)
/// * `weight_ptr` - Weight tensor (C, 1, K_h, K_w)
/// * `bias_ptr` - Optional bias tensor (C,)
/// * `output_ptr` - Output tensor (N, C, H_out, W_out)
///
/// # Safety
///
/// All pointers must be valid device memory with sufficient size.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_depthwise_conv2d(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    weight_ptr: u64,
    bias_ptr: Option<u64>,
    output_ptr: u64,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    output_h: usize,
    output_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
) -> Result<()> {
    let variant =
        depthwise_conv2d_variant(device_index, dtype, batch, channels, output_h, output_w);
    unsafe {
        launch_depthwise_conv2d_variant(
            context,
            stream,
            device_index,
            dtype,
            input_ptr,
            weight_ptr,
            bias_ptr,
            output_ptr,
            batch,
            channels,
            height,
            width,
            kernel_h,
            kernel_w,
            output_h,
            output_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            variant,
        )
    }
}

/// [`launch_depthwise_conv2d`] with the kernel named rather than chosen, so a
/// test can run both variants against the same shape.
///
/// `variant` must be one this shape supports: the blocked kernel does not
/// exist for the FP8 dtypes and needs `channels` and `batch` within the grid
/// limit.
///
/// # Safety
///
/// Same contract as [`launch_depthwise_conv2d`].
#[allow(clippy::too_many_arguments)]
unsafe fn launch_depthwise_conv2d_variant(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    weight_ptr: u64,
    bias_ptr: Option<u64>,
    output_ptr: u64,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    output_h: usize,
    output_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    variant: DepthwiseConv2dVariant,
) -> Result<()> {
    let total = batch * channels * output_h * output_w;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let ox_blocked = variant == DepthwiseConv2dVariant::Ox;
        let x_extent = output_w.div_ceil(DEPTHWISE_CONV2D_OX_BLOCK);
        let x_work = output_h.saturating_mul(x_extent);

        let base = if ox_blocked {
            "depthwise_conv2d_ox"
        } else {
            "depthwise_conv2d"
        };
        let module_name = if ox_blocked {
            DEPTHWISE_CONV2D_OX_MODULE
        } else {
            CONV_MODULE
        };
        let module = get_or_load_module(context, device_index, module_name)?;
        let func_name = kernel_name(base, dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let cfg = if ox_blocked {
            let grid = (
                (x_work as u32).div_ceil(CONV_BLOCK_THREADS),
                channels as u32,
                batch as u32,
            );
            launch_config(grid, (CONV_BLOCK_THREADS, 1, 1), 0)
        } else {
            let grid = elementwise_launch_config(total)?;
            launch_config(grid, (BLOCK_SIZE, 1, 1), 0)
        };

        let batch_u32 = batch as u32;
        let channels_u32 = channels as u32;
        let height_u32 = height as u32;
        let width_u32 = width as u32;
        let kernel_h_u32 = kernel_h as u32;
        let kernel_w_u32 = kernel_w as u32;
        let output_h_u32 = output_h as u32;
        let output_w_u32 = output_w as u32;
        let stride_h_u32 = stride_h as u32;
        let stride_w_u32 = stride_w as u32;
        let pad_h_u32 = pad_h as u32;
        let pad_w_u32 = pad_w as u32;
        let dilation_h_u32 = dilation_h as u32;
        let dilation_w_u32 = dilation_w as u32;
        let has_bias_u32: u32 = if bias_ptr.is_some() { 1 } else { 0 };
        let bias_ptr_val = bias_ptr.unwrap_or(0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&bias_ptr_val);
        builder.arg(&output_ptr);
        builder.arg(&batch_u32);
        builder.arg(&channels_u32);
        builder.arg(&height_u32);
        builder.arg(&width_u32);
        builder.arg(&kernel_h_u32);
        builder.arg(&kernel_w_u32);
        builder.arg(&output_h_u32);
        builder.arg(&output_w_u32);
        builder.arg(&stride_h_u32);
        builder.arg(&stride_w_u32);
        builder.arg(&pad_h_u32);
        builder.arg(&pad_w_u32);
        builder.arg(&dilation_h_u32);
        builder.arg(&dilation_w_u32);
        builder.arg(&has_bias_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA depthwise_conv2d kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::RandomOps;
    use crate::runtime::Runtime;
    use crate::runtime::cuda::{CudaClient, CudaRuntime};
    use crate::tensor::Tensor;

    fn setup() -> Option<CudaClient> {
        if !crate::runtime::cuda::is_cuda_available() {
            return None;
        }
        Some(CudaRuntime::default_client(&CudaDevice::new(0)))
    }

    /// One shape, both kernels a batch change can select between.
    ///
    /// `depthwise_conv2d_ox` blocks four output columns into one thread and
    /// reuses the overlapping taps; the flat kernel does one element per
    /// thread and reuses nothing. Neither reorders ONE element's sum, so the
    /// two must agree bit for bit. This is the reverse of
    /// `tests/cuda_conv_batch_invariance.rs`: that test pins the op across
    /// batch sizes, this one pins the two kernels the batch chooses between.
    #[test]
    fn flat_and_column_blocked_agree_bitwise() {
        let Some(client) = setup() else {
            return;
        };
        let (batch, channels, h, w, k) = (2usize, 6usize, 17usize, 23usize, 3usize);
        let (out_h, out_w) = (h, w);
        let input = client
            .rand(&[batch, channels, h, w], DType::F32)
            .expect("input");
        let weight = client
            .rand(&[channels, 1, k, k], DType::F32)
            .expect("weight");
        let bias = client.rand(&[channels], DType::F32).expect("bias");

        let run = |variant: DepthwiseConv2dVariant| -> Vec<f32> {
            let out = Tensor::<CudaRuntime>::empty(
                &[batch, channels, out_h, out_w],
                DType::F32,
                &client.device,
            )
            .expect("output");
            unsafe {
                launch_depthwise_conv2d_variant(
                    &client.context,
                    &client.stream,
                    client.device.index,
                    DType::F32,
                    input.ptr(),
                    weight.ptr(),
                    Some(bias.ptr()),
                    out.ptr(),
                    batch,
                    channels,
                    h,
                    w,
                    k,
                    k,
                    out_h,
                    out_w,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    variant,
                )
                .expect("launch");
            }
            out.to_vec()
        };

        let flat = run(DepthwiseConv2dVariant::Flat);
        let blocked = run(DepthwiseConv2dVariant::Ox);
        assert_eq!(flat.len(), blocked.len());
        for (i, (a, b)) in flat.iter().zip(&blocked).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {i}: flat {a} vs column-blocked {b}"
            );
        }
    }
}
