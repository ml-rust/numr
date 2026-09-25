//! conv1d CUDA kernel launcher.
//!
//! Three kernels serve one op: the scalar `conv1d`, the channel-blocked
//! `conv1d_oc4` and the position-blocked `conv1d_ox`. [`conv1d_variant`] picks
//! between them from the shape and the device. All three form one output
//! element's sum over `ic` ascending, then `kx` ascending, in the same
//! accumulator width, with bias added last, so the choice moves no bits.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    launch_config,
};
use super::tuning::{
    CONV_BLOCK_THREADS, CONV_MODULE, CONV1D_BLOCK_CANDIDATES, CONV1D_BLOCK_MAX, CONV1D_OC_BLOCK,
    CONV1D_OX_BLOCK, CONV1D_OX_MIN_OUTPUT_LENGTH, CONV1D_OX_MIN_WAVES, CONV1D_OX_MODULE,
    CUDA_MAX_GRID_YZ, fills_device, position_block_width,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Device;
use crate::runtime::cuda::CudaDevice;
use crate::runtime::cuda::GuardedStream;

/// Which conv1d kernel a launch runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Conv1dVariant {
    /// `conv1d`: one output element per thread.
    Scalar,
    /// `conv1d_oc4`: [`CONV1D_OC_BLOCK`] output channels per thread.
    Oc4,
    /// `conv1d_ox`: [`CONV1D_OX_BLOCK`] output positions per thread.
    Ox,
}

/// The kernel this shape runs on.
///
/// Register-block over output channels when a group holds at least
/// [`CONV1D_OC_BLOCK`] of them. Depthwise (`c_out_per_group == 1`) and other
/// narrow-group shapes fall back to `conv1d_ox` when the row is wide enough
/// and the blocked grid still fills the device, else to the scalar kernel.
fn conv1d_variant(
    device_index: usize,
    dtype: DType,
    batch: usize,
    c_out: usize,
    output_length: usize,
    groups: usize,
) -> Conv1dVariant {
    let is_fp8 = matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2);
    if is_fp8 {
        return Conv1dVariant::Scalar;
    }
    if c_out.checked_div(groups).unwrap_or(0) >= CONV1D_OC_BLOCK {
        return Conv1dVariant::Oc4;
    }

    // Threads the position-blocked grid would launch, and the wave it must
    // fill.
    let compute_units = CudaDevice::new(device_index).profile().compute_units as usize;
    let ox_threads = batch
        .saturating_mul(c_out)
        .saturating_mul(output_length.div_ceil(CONV1D_OX_BLOCK));
    if output_length >= CONV1D_OX_MIN_OUTPUT_LENGTH
        && fills_device(ox_threads, compute_units, CONV1D_OX_MIN_WAVES)
    {
        Conv1dVariant::Ox
    } else {
        Conv1dVariant::Scalar
    }
}

/// Launch conv1d kernel.
///
/// Performs 1D convolution with optional groups support.
///
/// # Arguments
///
/// * `input_ptr` - Input tensor (N, C_in, L)
/// * `weight_ptr` - Weight tensor (C_out, C_in/groups, K)
/// * `bias_ptr` - Optional bias tensor (C_out,)
/// * `output_ptr` - Output tensor (N, C_out, L_out)
///
/// # Safety
///
/// All pointers must be valid device memory with sufficient size.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_conv1d(
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
    let variant = conv1d_variant(device_index, dtype, batch, c_out, output_length, groups);
    unsafe {
        launch_conv1d_variant(
            context,
            stream,
            device_index,
            dtype,
            input_ptr,
            weight_ptr,
            bias_ptr,
            output_ptr,
            batch,
            c_in,
            length,
            c_out,
            kernel_size,
            output_length,
            stride,
            padding,
            dilation,
            groups,
            variant,
        )
    }
}

/// [`launch_conv1d`] with the kernel named rather than chosen, so a test can
/// run two variants against the same shape.
///
/// `variant` must be one this shape supports: `Oc4` needs
/// `c_out / groups >= CONV1D_OC_BLOCK`, and neither blocked kernel exists for
/// the FP8 dtypes.
///
/// # Safety
///
/// Same contract as [`launch_conv1d`].
#[allow(clippy::too_many_arguments)]
unsafe fn launch_conv1d_variant(
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
    variant: Conv1dVariant,
) -> Result<()> {
    let total = batch * c_out * output_length;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let c_out_per_group = c_out.checked_div(groups).unwrap_or(0);
        let c_in_per_group = c_in.checked_div(groups).unwrap_or(0);
        let is_fp8 = matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2);
        let oc_blocked = variant == Conv1dVariant::Oc4;
        let ox_blocked = variant == Conv1dVariant::Ox;

        let base = if oc_blocked {
            "conv1d_oc4"
        } else if ox_blocked {
            "conv1d_ox"
        } else {
            "conv1d"
        };
        let module_name = if ox_blocked {
            CONV1D_OX_MODULE
        } else {
            CONV_MODULE
        };
        let module = get_or_load_module(context, device_index, module_name)?;
        let func_name = kernel_name(base, dtype);
        let func = get_kernel_function(&module, &func_name)?;

        // The FP8 conv1d kernel keeps its own legacy flat launch over a linear
        // index; only the macro-generated float kernels take the (ox, slot,
        // batch) grid that removes the per-thread integer division.
        let three_d_grid = !is_fp8;

        let cfg = if three_d_grid {
            // conv1d_ox packs CONV1D_OX_BLOCK output positions per thread, so
            // the x axis walks blocks-of-CONV1D_OX_BLOCK, not raw positions.
            let x_extent = if ox_blocked {
                output_length.div_ceil(CONV1D_OX_BLOCK)
            } else {
                output_length
            };

            // oc4 keeps the warp floor; the scalar and ox kernels may go
            // narrower.
            let block_x = if oc_blocked {
                CONV1D_BLOCK_CANDIDATES
                    .into_iter()
                    .find(|&w| x_extent <= w as usize)
                    .unwrap_or(CONV1D_BLOCK_MAX)
            } else {
                position_block_width(x_extent)
            };
            let block_y = (CONV_BLOCK_THREADS / block_x).max(1);

            // One slot per output channel, or per chunk of CONV1D_OC_BLOCK
            // channels when the register-blocked kernel runs. Chunking is per
            // group so the channels a thread blocks over share a c_in range.
            let slots = if oc_blocked {
                groups * c_out_per_group.div_ceil(CONV1D_OC_BLOCK)
            } else {
                c_out
            };
            let grid_y = slots.div_ceil(block_y as usize);

            if grid_y > CUDA_MAX_GRID_YZ || batch > CUDA_MAX_GRID_YZ {
                return Err(Error::Internal(format!(
                    "CUDA conv1d: grid y={} z={} exceed the grid limit of {}",
                    grid_y, batch, CUDA_MAX_GRID_YZ
                )));
            }

            let grid = (
                (x_extent as u32).div_ceil(block_x),
                grid_y as u32,
                batch as u32,
            );
            launch_config(grid, (block_x, block_y, 1), 0)
        } else {
            let grid = elementwise_launch_config(total)?;
            launch_config(grid, (BLOCK_SIZE, 1, 1), 0)
        };

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
        let c_in_per_group_u32 = c_in_per_group as u32;
        let c_out_per_group_u32 = c_out_per_group as u32;
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
        // The FP8 conv1d kernel keeps its own legacy flat signature (see
        // `three_d_grid` above) and was never given these two host-computed
        // params, so only the macro-generated kernels receive them.
        if three_d_grid {
            builder.arg(&c_in_per_group_u32);
            builder.arg(&c_out_per_group_u32);
        }
        builder.arg(&has_bias_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA conv1d kernel launch failed: {:?}", e)))?;

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
    /// `conv1d_ox` blocks four output positions into one thread and reuses the
    /// overlapping taps; the scalar kernel does one position per thread and
    /// reuses nothing. Neither reorders ONE element's sum, so the two must
    /// agree bit for bit. This is the reverse of
    /// `tests/cuda_conv_batch_invariance.rs`: that test pins the op across
    /// batch sizes, this one pins the two kernels the batch chooses between.
    #[test]
    fn scalar_and_position_blocked_agree_bitwise() {
        let Some(client) = setup() else {
            return;
        };
        // Depthwise, so `conv1d_oc4` is out and the choice is scalar vs ox.
        let (batch, channels, length, k) = (2usize, 8usize, 64usize, 3usize);
        let output_length = length - k + 1;
        let input = client
            .rand(&[batch, channels, length], DType::F32)
            .expect("input");
        let weight = client.rand(&[channels, 1, k], DType::F32).expect("weight");
        let bias = client.rand(&[channels], DType::F32).expect("bias");

        let run = |variant: Conv1dVariant| -> Vec<f32> {
            let out = Tensor::<CudaRuntime>::empty(
                &[batch, channels, output_length],
                DType::F32,
                &client.device,
            )
            .expect("output");
            unsafe {
                launch_conv1d_variant(
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
                    length,
                    channels,
                    k,
                    output_length,
                    1,
                    0,
                    1,
                    channels,
                    variant,
                )
                .expect("launch");
            }
            out.to_vec()
        };

        let scalar = run(Conv1dVariant::Scalar);
        let blocked = run(Conv1dVariant::Ox);
        assert_eq!(scalar.len(), blocked.len());
        for (i, (a, b)) in scalar.iter().zip(&blocked).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {i}: scalar {a} vs position-blocked {b}"
            );
        }
    }
}
