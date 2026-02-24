//! Reduction WGSL kernel launchers. F32, I32, U32.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

const REDUCE_F32_SHADER: &str = include_str!("reduce.wgsl");
const REDUCE_I32_SHADER: &str = include_str!("reduce_i32.wgsl");
const REDUCE_U32_SHADER: &str = include_str!("reduce_u32.wgsl");

// ============================================================================
// Single-Dimension Reduction
// ============================================================================

/// Launch a reduction operation along a single dimension. F32, I32, U32.
///
/// Supported ops: "sum", "mean" (F32 only), "max", "min", "prod", "any", "all"
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
    // mean is F32-only
    if op == "mean" && dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }

    let (module_key, shader, suffix) = match dtype {
        DType::F32 => ("reduce_f32", REDUCE_F32_SHADER, "f32"),
        DType::I32 => ("reduce_i32", REDUCE_I32_SHADER, "i32"),
        DType::U32 => ("reduce_u32", REDUCE_U32_SHADER, "u32"),
        _ => return Err(Error::UnsupportedDType { dtype, op }),
    };

    let entry_point: String = match op {
        "sum" | "mean" | "max" | "min" | "prod" | "any" | "all" => {
            format!("reduce_{}_{}", op, suffix)
        }
        _ => return Err(Error::Internal(format!("Unknown reduce op: {}", op))),
    };

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(op) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(numel_out as u32, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Full Reduction (all elements to single value)
// ============================================================================

/// Launch a full reduction kernel (reduce all elements). F32, I32, U32.
///
/// Supported ops: "sum", "max", "min", "prod"
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
    let (module_key, shader, suffix) = match dtype {
        DType::F32 => ("reduce_f32", REDUCE_F32_SHADER, "f32"),
        DType::I32 => ("reduce_i32", REDUCE_I32_SHADER, "i32"),
        DType::U32 => ("reduce_u32", REDUCE_U32_SHADER, "u32"),
        _ => return Err(Error::UnsupportedDType { dtype, op }),
    };

    let entry_point: String = match op {
        "sum" | "max" | "min" | "prod" => format!("full_reduce_{}_{}", op, suffix),
        _ => return Err(Error::Internal(format!("Unknown full reduce op: {}", op))),
    };

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(op) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Argmax / Argmin
// ============================================================================

/// Launch argmax/argmin kernel. F32, I32, U32.
///
/// Supported ops: "argmax", "argmin"
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
    let (module_key, shader, suffix) = match dtype {
        DType::F32 => ("reduce_f32", REDUCE_F32_SHADER, "f32"),
        DType::I32 => ("reduce_i32", REDUCE_I32_SHADER, "i32"),
        DType::U32 => ("reduce_u32", REDUCE_U32_SHADER, "u32"),
        _ => return Err(Error::UnsupportedDType { dtype, op }),
    };

    let entry_point: String = match op {
        "argmax" | "argmin" => format!("{}_{}", op, suffix),
        _ => return Err(Error::Internal(format!("Unknown argreduce op: {}", op))),
    };

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(op) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(numel_out as u32, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Softmax
// ============================================================================

/// Launch softmax kernel. F32 only.
pub fn launch_softmax_op(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "softmax",
        });
    }

    let module = cache.get_or_create_module("reduce_f32", REDUCE_F32_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("reduce_f32", "softmax_f32", &module, &layout);
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
        pass.dispatch_workgroups(batch_size as u32, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
