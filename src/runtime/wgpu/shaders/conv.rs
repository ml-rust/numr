//! Convolution WGSL kernel launchers
//!
//! Provides launchers for convolution operations:
//! - 1D convolution (conv1d)
//! - 2D convolution (conv2d)
//! - Depthwise 2D convolution (depthwise_conv2d)
//!
//! All operations run entirely on GPU with no CPU fallback.

use wgpu::{Buffer, Queue};

use super::generator::{
    generate_conv1d_shader, generate_conv2d_shader, generate_depthwise_conv2d_shader,
};
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Helper Macros
// ============================================================================

macro_rules! check_dtype_float {
    ($dtype:expr, $op:expr) => {
        if $dtype != DType::F32 && $dtype != DType::F16 {
            return Err(Error::UnsupportedDType {
                dtype: $dtype,
                op: $op,
            });
        }
    };
}

/// Get static kernel name for convolution operations.
fn kernel_name(op: &'static str, dtype: DType) -> Result<&'static str> {
    match (op, dtype) {
        ("conv1d", DType::F32) => Ok("conv1d_f32"),
        ("conv1d", DType::F16) => Ok("conv1d_f16"),
        ("conv2d", DType::F32) => Ok("conv2d_f32"),
        ("conv2d", DType::F16) => Ok("conv2d_f16"),
        ("depthwise_conv2d", DType::F32) => Ok("depthwise_conv2d_f32"),
        ("depthwise_conv2d", DType::F16) => Ok("depthwise_conv2d_f16"),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

// ============================================================================
// Conv1d
// ============================================================================

/// Launch conv1d kernel.
///
/// Performs 1D convolution with optional groups support.
///
/// # Arguments
///
/// * `input` - Input tensor (N, C_in, L)
/// * `weight` - Weight tensor (C_out, C_in/groups, K)
/// * `bias` - Optional bias tensor (C_out)
/// * `output` - Output tensor (N, C_out, L_out)
/// * `params_buffer` - Uniform buffer with Conv1dParams
/// * `total_output` - Total number of output elements
pub fn launch_conv1d(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    weight: &Buffer,
    bias: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_output: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_float!(dtype, "conv1d");

    let name = kernel_name("conv1d", dtype)?;
    let shader_source = generate_conv1d_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[input, weight, bias, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("conv1d"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("conv1d"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_output), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Conv2d
// ============================================================================

/// Launch conv2d kernel.
///
/// Performs 2D convolution with optional groups support.
///
/// # Arguments
///
/// * `input` - Input tensor (N, C_in, H, W)
/// * `weight` - Weight tensor (C_out, C_in/groups, K_h, K_w)
/// * `bias` - Optional bias tensor (C_out)
/// * `output` - Output tensor (N, C_out, H_out, W_out)
/// * `params_buffer` - Uniform buffer with Conv2dParams
/// * `total_output` - Total number of output elements
pub fn launch_conv2d(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    weight: &Buffer,
    bias: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_output: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_float!(dtype, "conv2d");

    let name = kernel_name("conv2d", dtype)?;
    let shader_source = generate_conv2d_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[input, weight, bias, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("conv2d"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("conv2d"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_output), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Depthwise Conv2d
// ============================================================================

/// Launch depthwise conv2d kernel.
///
/// Performs depthwise 2D convolution where each channel is convolved independently.
///
/// # Arguments
///
/// * `input` - Input tensor (N, C, H, W)
/// * `weight` - Weight tensor (C, 1, K_h, K_w)
/// * `bias` - Optional bias tensor (C)
/// * `output` - Output tensor (N, C, H_out, W_out)
/// * `params_buffer` - Uniform buffer with DepthwiseConv2dParams
/// * `total_output` - Total number of output elements
pub fn launch_depthwise_conv2d(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    weight: &Buffer,
    bias: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_output: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_float!(dtype, "depthwise_conv2d");

    let name = kernel_name("depthwise_conv2d", dtype)?;
    let shader_source = generate_depthwise_conv2d_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[input, weight, bias, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("depthwise_conv2d"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("depthwise_conv2d"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_output), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
