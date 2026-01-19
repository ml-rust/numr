//! Element-wise WGSL kernel launchers
//!
//! Provides launchers for element-wise operations including:
//! - Binary operations (add, sub, mul, div, pow, max, min)
//! - Unary operations (neg, abs, sqrt, exp, log, sin, cos, tan, tanh, etc.)
//! - Scalar operations (add_scalar, sub_scalar, mul_scalar, div_scalar, pow_scalar)
//! - Comparison operations (eq, ne, lt, le, gt, ge)
//! - Activation functions (relu, sigmoid, silu, gelu)
//! - Utility operations (clamp, isnan, isinf, where)
//!
//! All operations run entirely on GPU with no CPU fallback.

use wgpu::{Buffer, Queue};

use super::elementwise_wgsl::ELEMENTWISE_SHADER;
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
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
// Binary Operations
// ============================================================================

/// Launch a binary element-wise operation kernel.
///
/// Computes `out[i] = a[i] op b[i]` for all elements.
pub fn launch_binary_op(
    cache: &PipelineCache,
    queue: &Queue,
    op: &'static str,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, op);

    let entry_point = match op {
        "add" => "add_f32",
        "sub" => "sub_f32",
        "mul" => "mul_f32",
        "div" => "div_f32",
        "pow" => "pow_f32",
        "max" | "maximum" => "max_f32",
        "min" | "minimum" => "min_f32",
        _ => return Err(Error::Internal(format!("Unknown binary op: {}", op))),
    };

    let module = cache.get_or_create_module("elementwise", ELEMENTWISE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("elementwise", entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, out, params_buffer]);

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
// Unary Operations
// ============================================================================

/// Launch a unary element-wise operation kernel.
///
/// Computes `out[i] = op(a[i])` for all elements.
pub fn launch_unary_op(
    cache: &PipelineCache,
    queue: &Queue,
    op: &'static str,
    a: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, op);

    let entry_point = match op {
        "neg" => "neg_f32",
        "abs" => "abs_f32",
        "sqrt" => "sqrt_f32",
        "exp" => "exp_f32",
        "log" => "log_f32",
        "sin" => "sin_f32",
        "cos" => "cos_f32",
        "tan" => "tan_f32",
        "tanh" => "tanh_f32",
        "recip" => "recip_f32",
        "square" => "square_f32",
        "floor" => "floor_f32",
        "ceil" => "ceil_f32",
        "round" => "round_f32",
        "sign" => "sign_f32",
        "relu" => "relu_f32",
        "sigmoid" => "sigmoid_f32",
        "silu" => "silu_f32",
        "gelu" => "gelu_f32",
        "isnan" => "isnan_f32",
        "isinf" => "isinf_f32",
        _ => return Err(Error::Internal(format!("Unknown unary op: {}", op))),
    };

    let module = cache.get_or_create_module("elementwise", ELEMENTWISE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("elementwise", entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, out, params_buffer]);

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
// Scalar Operations
// ============================================================================

/// Launch a scalar element-wise operation kernel.
///
/// Computes `out[i] = a[i] op scalar` for all elements.
pub fn launch_scalar_op(
    cache: &PipelineCache,
    queue: &Queue,
    op: &'static str,
    a: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, op);

    let entry_point = match op {
        "add_scalar" => "add_scalar_f32",
        "sub_scalar" => "sub_scalar_f32",
        "mul_scalar" => "mul_scalar_f32",
        "div_scalar" => "div_scalar_f32",
        "pow_scalar" => "pow_scalar_f32",
        _ => return Err(Error::Internal(format!("Unknown scalar op: {}", op))),
    };

    let module = cache.get_or_create_module("elementwise", ELEMENTWISE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("elementwise", entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, out, params_buffer]);

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
// Comparison Operations
// ============================================================================

/// Launch a comparison element-wise operation kernel.
///
/// Computes `out[i] = (a[i] op b[i]) ? 1.0 : 0.0` for all elements.
pub fn launch_compare_op(
    cache: &PipelineCache,
    queue: &Queue,
    op: &'static str,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, op);

    let entry_point = match op {
        "eq" => "eq_f32",
        "ne" => "ne_f32",
        "lt" => "lt_f32",
        "le" => "le_f32",
        "gt" => "gt_f32",
        "ge" => "ge_f32",
        _ => return Err(Error::Internal(format!("Unknown compare op: {}", op))),
    };

    let module = cache.get_or_create_module("elementwise", ELEMENTWISE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("elementwise", entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, out, params_buffer]);

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
// Clamp Operation
// ============================================================================

/// Launch clamp operation kernel.
///
/// Computes `out[i] = clamp(a[i], min_val, max_val)` for all elements.
pub fn launch_clamp_op(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "clamp");

    let module = cache.get_or_create_module("elementwise", ELEMENTWISE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("elementwise", "clamp_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("clamp"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("clamp"),
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
// Where Conditional Operation
// ============================================================================

/// Launch where conditional operation kernel.
///
/// Computes `out[i] = cond[i] ? x[i] : y[i]` for all elements.
pub fn launch_where_op(
    cache: &PipelineCache,
    queue: &Queue,
    cond: &Buffer,
    x: &Buffer,
    y: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "where");

    let module = cache.get_or_create_module("elementwise", ELEMENTWISE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("elementwise", "where_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[cond, x, y, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("where"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("where"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
