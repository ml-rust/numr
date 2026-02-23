//! Element-wise WGSL kernel launchers
//!
//! All operations are F32-only. WebGPU is a 32-bit compute backend by design.
//! For other dtypes use the CPU or CUDA backends.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Static Shader Sources
// ============================================================================

const BINARY_SHADER: &str = include_str!("binary.wgsl");
const BINARY_BROADCAST_SHADER: &str = include_str!("binary_broadcast.wgsl");
const UNARY_SHADER: &str = include_str!("unary.wgsl");
const SCALAR_SHADER: &str = include_str!("scalar.wgsl");
const COMPARE_SHADER: &str = include_str!("compare.wgsl");

const CAST_F32_TO_I32_SHADER: &str = include_str!("cast_f32_to_i32.wgsl");
const CAST_F32_TO_U32_SHADER: &str = include_str!("cast_f32_to_u32.wgsl");
const CAST_I32_TO_F32_SHADER: &str = include_str!("cast_i32_to_f32.wgsl");
const CAST_I32_TO_U32_SHADER: &str = include_str!("cast_i32_to_u32.wgsl");
const CAST_U32_TO_F32_SHADER: &str = include_str!("cast_u32_to_f32.wgsl");
const CAST_U32_TO_I32_SHADER: &str = include_str!("cast_u32_to_i32.wgsl");

// ============================================================================
// Binary Operations
// ============================================================================

/// Launch a binary element-wise operation: `out[i] = a[i] op b[i]`. F32 only.
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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }

    let op_name = match op {
        "maximum" => "max",
        "minimum" => "min",
        _ => op,
    };

    let entry_point: &'static str = match op_name {
        "add" => "add_f32",
        "sub" => "sub_f32",
        "mul" => "mul_f32",
        "div" => "div_f32",
        "max" => "max_f32",
        "min" => "min_f32",
        "pow" => "pow_f32",
        "atan2" => "atan2_f32",
        _ => return Err(Error::Internal(format!("Unknown binary op: {}", op_name))),
    };

    let module = cache.get_or_create_module("binary_f32", BINARY_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("binary_f32", entry_point, &module, &layout);
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

/// Launch a broadcast binary operation. F32 only.
#[allow(clippy::too_many_arguments)]
pub fn launch_broadcast_binary_op(
    cache: &PipelineCache,
    queue: &Queue,
    op: &'static str,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    a_strides: &Buffer,
    b_strides: &Buffer,
    out_strides: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }

    let op_name = match op {
        "maximum" => "max",
        "minimum" => "min",
        _ => op,
    };

    let entry_point: &'static str = match op_name {
        "add" => "broadcast_add_f32",
        "sub" => "broadcast_sub_f32",
        "mul" => "broadcast_mul_f32",
        "div" => "broadcast_div_f32",
        "max" => "broadcast_max_f32",
        "min" => "broadcast_min_f32",
        "pow" => "broadcast_pow_f32",
        _ => {
            return Err(Error::Internal(format!(
                "Unknown broadcast binary op: {}",
                op_name
            )));
        }
    };

    let module = cache.get_or_create_module("binary_broadcast_f32", BINARY_BROADCAST_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("binary_broadcast_f32", entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(
        &layout,
        &[a, b, out, a_strides, b_strides, out_strides, params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("broadcast_binary"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("broadcast_binary"),
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

/// Launch a unary operation: `out[i] = op(a[i])`. F32 only.
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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }

    let entry_point: &'static str = match op {
        "neg" => "neg_f32",
        "abs" => "abs_f32",
        "sqrt" => "sqrt_f32",
        "exp" => "exp_f32",
        "log" => "log_f32",
        "sin" => "sin_f32",
        "cos" => "cos_f32",
        "tan" => "tan_f32",
        "atan" => "atan_f32",
        "tanh" => "tanh_f32",
        "recip" => "recip_f32",
        "floor" => "floor_f32",
        "ceil" => "ceil_f32",
        "round" => "round_f32",
        "trunc" => "trunc_f32",
        "rsqrt" => "rsqrt_f32",
        "cbrt" => "cbrt_f32",
        "exp2" => "exp2_f32",
        "expm1" => "expm1_f32",
        "log2" => "log2_f32",
        "log10" => "log10_f32",
        "log1p" => "log1p_f32",
        "asin" => "asin_f32",
        "acos" => "acos_f32",
        "sinh" => "sinh_f32",
        "cosh" => "cosh_f32",
        "asinh" => "asinh_f32",
        "acosh" => "acosh_f32",
        "atanh" => "atanh_f32",
        "square" => "square_f32",
        "sign" => "sign_f32",
        "relu" => "relu_f32",
        "sigmoid" => "sigmoid_f32",
        "silu" => "silu_f32",
        "gelu" => "gelu_f32",
        "isnan" => "isnan_f32",
        "isinf" => "isinf_f32",
        _ => return Err(Error::Internal(format!("Unknown unary op: {}", op))),
    };

    let module = cache.get_or_create_module("unary_f32", UNARY_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("unary_f32", entry_point, &module, &layout);
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

/// Launch a scalar operation: `out[i] = a[i] op scalar`. F32 only.
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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }

    let entry_point: &'static str = match op {
        "add_scalar" => "add_scalar_f32",
        "sub_scalar" => "sub_scalar_f32",
        "rsub_scalar" => "rsub_scalar_f32",
        "mul_scalar" => "mul_scalar_f32",
        "div_scalar" => "div_scalar_f32",
        "pow_scalar" => "pow_scalar_f32",
        _ => return Err(Error::Internal(format!("Unknown scalar op: {}", op))),
    };

    let module = cache.get_or_create_module("scalar_f32", SCALAR_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("scalar_f32", entry_point, &module, &layout);
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

/// Launch a comparison operation: `out[i] = (a[i] op b[i]) ? 1.0 : 0.0`. F32 only.
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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }

    let entry_point: &'static str = match op {
        "eq" => "eq_f32",
        "ne" => "ne_f32",
        "lt" => "lt_f32",
        "le" => "le_f32",
        "gt" => "gt_f32",
        "ge" => "ge_f32",
        _ => return Err(Error::Internal(format!("Unknown compare op: {}", op))),
    };

    let module = cache.get_or_create_module("compare_f32", COMPARE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("compare_f32", entry_point, &module, &layout);
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
// Cast Operations
// ============================================================================

/// Launch a cast operation: `out[i] = DstType(a[i])`. Supports F32 ↔ I32 ↔ U32.
pub fn launch_cast_op(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    src_dtype: DType,
    dst_dtype: DType,
) -> Result<()> {
    if src_dtype == dst_dtype {
        return Ok(());
    }

    let (module_name, entry_point, shader_source): (&'static str, &'static str, &'static str) =
        match (src_dtype, dst_dtype) {
            (DType::F32, DType::I32) => ("cast_f32_i32", "cast_f32_to_i32", CAST_F32_TO_I32_SHADER),
            (DType::F32, DType::U32) => ("cast_f32_u32", "cast_f32_to_u32", CAST_F32_TO_U32_SHADER),
            (DType::I32, DType::F32) => ("cast_i32_f32", "cast_i32_to_f32", CAST_I32_TO_F32_SHADER),
            (DType::I32, DType::U32) => ("cast_i32_u32", "cast_i32_to_u32", CAST_I32_TO_U32_SHADER),
            (DType::U32, DType::F32) => ("cast_u32_f32", "cast_u32_to_f32", CAST_U32_TO_F32_SHADER),
            (DType::U32, DType::I32) => ("cast_u32_i32", "cast_u32_to_i32", CAST_U32_TO_I32_SHADER),
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype: src_dtype,
                    op: "cast",
                });
            }
        };

    let module = cache.get_or_create_module(module_name, shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_name, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[a, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cast"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cast"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
