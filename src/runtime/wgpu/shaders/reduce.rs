//! Reduction WGSL kernel launchers
//!
//! Provides launchers for reduction operations including:
//! - Sum, Mean, Max, Min reductions along specified dimensions
//! - Argmax, Argmin (returns indices)
//! - Softmax (numerically stable)
//!
//! All operations run entirely on GPU with no CPU fallback.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use super::reduce_wgsl::REDUCE_SHADER;
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Helper Macros
// ============================================================================

macro_rules! check_dtype_f32 {
    ($dtype:expr, $op:expr) => {
        if $dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype: $dtype,
                op: $op,
            });
        }
    };
}

// ============================================================================
// Single-Dimension Reduction
// ============================================================================

/// Launch a reduction operation kernel along a single dimension.
///
/// Parameters:
/// - reduce_size: Size of the dimension being reduced
/// - outer_size: Product of dimensions before the reduce dimension
/// - inner_size: Product of dimensions after the reduce dimension
pub fn launch_reduce_op(
    cache: &PipelineCache,
    queue: &Queue,
    op: &'static str,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    numel_out: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, op);

    let entry_point = match op {
        "sum" => "reduce_sum_f32",
        "mean" => "reduce_mean_f32",
        "max" => "reduce_max_f32",
        "min" => "reduce_min_f32",
        _ => return Err(Error::Internal(format!("Unknown reduce op: {}", op))),
    };

    let module = cache.get_or_create_module("reduce", REDUCE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("reduce", entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(op),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per output element
        pass.dispatch_workgroups(numel_out as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Full Reduction (all elements to single value)
// ============================================================================

/// Launch a full reduction operation kernel.
///
/// Reduces all elements to a single value using two-pass reduction.
pub fn launch_full_reduce_op(
    cache: &PipelineCache,
    queue: &Queue,
    op: &'static str,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, op);

    let entry_point = match op {
        "sum" => "full_reduce_sum_f32",
        "max" => "full_reduce_max_f32",
        "min" => "full_reduce_min_f32",
        _ => return Err(Error::Internal(format!("Unknown full reduce op: {}", op))),
    };

    let module = cache.get_or_create_module("reduce", REDUCE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("reduce", entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(op),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // Use enough workgroups to cover all elements
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Argmax / Argmin
// ============================================================================

/// Launch argmax/argmin operation kernel.
///
/// Returns indices of max/min values along specified dimension.
pub fn launch_argreduce_op(
    cache: &PipelineCache,
    queue: &Queue,
    op: &'static str,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    numel_out: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, op);

    let entry_point = match op {
        "argmax" => "argmax_f32",
        "argmin" => "argmin_f32",
        _ => return Err(Error::Internal(format!("Unknown argreduce op: {}", op))),
    };

    let module = cache.get_or_create_module("reduce", REDUCE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("reduce", entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(op),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per output element
        pass.dispatch_workgroups(numel_out as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Softmax
// ============================================================================

/// Launch softmax operation kernel.
///
/// Computes numerically stable softmax over the last dimension.
pub fn launch_softmax_op(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "softmax");

    let module = cache.get_or_create_module("reduce", REDUCE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("reduce", "softmax_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("softmax"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("softmax"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per batch element
        pass.dispatch_workgroups(batch_size as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
