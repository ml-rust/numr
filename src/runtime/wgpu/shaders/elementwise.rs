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
//! Multi-dtype support: F32, I32, U32 (F16 requires shader-f16 extension)
//! All operations run entirely on GPU with no CPU fallback.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use wgpu::{Buffer, Queue};

use super::dtype_support;
use super::elementwise_wgsl::ELEMENTWISE_SHADER;
use super::generator::{
    dtype_suffix, generate_binary_shader, generate_cast_shader, generate_compare_shader,
    generate_scalar_shader, generate_unary_shader,
};
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Shader Module Cache
// ============================================================================

/// Cache for leaked shader references (leaked once per dtype+op_type combination)
/// Key: (DType, operation_type), Value: &'static str to leaked shader source
static SHADER_CACHE: OnceLock<RwLock<HashMap<(DType, &'static str), &'static str>>> =
    OnceLock::new();

/// Cache for leaked module key references
static MODULE_KEY_CACHE: OnceLock<RwLock<HashMap<(DType, &'static str), &'static str>>> =
    OnceLock::new();

/// Get or generate shader for a specific dtype and operation type.
/// Generates shader once, leaks it once, caches the leaked reference.
/// Subsequent calls return the cached &'static str without leaking.
fn get_or_leak_shader(dtype: DType, op_type: &'static str) -> Result<&'static str> {
    let cache = SHADER_CACHE.get_or_init(|| RwLock::new(HashMap::new()));

    // Check if already cached
    {
        let read_guard = cache.read().unwrap();
        if let Some(&shader_ref) = read_guard.get(&(dtype, op_type)) {
            return Ok(shader_ref);
        }
    }

    // Generate shader based on operation type
    let shader = match op_type {
        "binary" => generate_binary_shader(dtype)?,
        "unary" => generate_unary_shader(dtype)?,
        "scalar" => generate_scalar_shader(dtype)?,
        "compare" => generate_compare_shader(dtype)?,
        _ => return Err(Error::Internal(format!("Unknown op type: {}", op_type))),
    };

    // Leak ONCE and cache the reference
    let leaked: &'static str = Box::leak(shader.into_boxed_str());

    let mut write_guard = cache.write().unwrap();
    write_guard.insert((dtype, op_type), leaked);

    Ok(leaked)
}

/// Get the module key for a dtype and operation type.
/// Generates key once, leaks it once, caches the leaked reference.
fn get_or_leak_module_key(dtype: DType, op_type: &'static str) -> Result<&'static str> {
    let cache = MODULE_KEY_CACHE.get_or_init(|| RwLock::new(HashMap::new()));

    // Check if already cached
    {
        let read_guard = cache.read().unwrap();
        if let Some(&key_ref) = read_guard.get(&(dtype, op_type)) {
            return Ok(key_ref);
        }
    }

    // Generate module key
    let suffix = dtype_suffix(dtype)?;
    let key = format!("{}_{}", op_type, suffix);

    // Leak ONCE and cache the reference
    let leaked: &'static str = Box::leak(key.into_boxed_str());

    let mut write_guard = cache.write().unwrap();
    write_guard.insert((dtype, op_type), leaked);

    Ok(leaked)
}

/// Cache for leaked entry point references
static ENTRY_POINT_CACHE: OnceLock<RwLock<HashMap<(String, DType), &'static str>>> =
    OnceLock::new();

/// Get entry point name for an operation.
/// Generates once per (op, dtype), leaks once, caches the leaked reference.
fn get_or_leak_entry_point(op: &str, dtype: DType) -> Result<&'static str> {
    let cache = ENTRY_POINT_CACHE.get_or_init(|| RwLock::new(HashMap::new()));

    let key = (op.to_string(), dtype);

    // Check if already cached
    {
        let read_guard = cache.read().unwrap();
        if let Some(&entry_ref) = read_guard.get(&key) {
            return Ok(entry_ref);
        }
    }

    // Generate entry point
    let suffix = dtype_suffix(dtype)?;
    let entry = format!("{}_{}", op, suffix);

    // Leak ONCE and cache the reference
    let leaked: &'static str = Box::leak(entry.into_boxed_str());

    let mut write_guard = cache.write().unwrap();
    write_guard.insert(key, leaked);

    Ok(leaked)
}

// ============================================================================
// Binary Operations
// ============================================================================

/// Launch a binary element-wise operation kernel.
///
/// Computes `out[i] = a[i] op b[i]` for all elements.
///
/// Supports F32, I32, U32 dtypes.
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
    // Validate dtype support for this operation
    dtype_support::check_binary_dtype_support(op, dtype)?;

    // Normalize operation name
    let op_name = match op {
        "maximum" => "max",
        "minimum" => "min",
        _ => op,
    };

    // Get entry point name based on dtype (cached, leaked once per op+dtype)
    let entry_point = get_or_leak_entry_point(op_name, dtype)?;

    // Use F32 shader for backward compatibility, or dtype-specific for I32/U32
    let (module_name, shader_source): (&str, &str) = if dtype == DType::F32 {
        ("elementwise", ELEMENTWISE_SHADER)
    } else {
        // For I32/U32, get cached shader and module key (leaked once per dtype+op_type)
        let shader = get_or_leak_shader(dtype, "binary")?;
        let module_key = get_or_leak_module_key(dtype, "binary")?;
        (module_key, shader)
    };

    let module = cache.get_or_create_module(module_name, shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(module_name, entry_point, &module, &layout);

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
///
/// Supports F32, I32, U32 dtypes (operation-dependent).
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
    // Validate dtype support for this operation
    dtype_support::check_unary_dtype_support(op, dtype)?;

    // Get entry point name based on dtype (cached, leaked once per op+dtype)
    let entry_point = get_or_leak_entry_point(op, dtype)?;

    // Use F32 shader for backward compatibility, or dtype-specific for I32/U32
    let (module_name, shader_source): (&str, &str) = if dtype == DType::F32 {
        ("elementwise", ELEMENTWISE_SHADER)
    } else {
        // For I32/U32, get cached shader and module key (leaked once per dtype+op_type)
        let shader = get_or_leak_shader(dtype, "unary")?;
        let module_key = get_or_leak_module_key(dtype, "unary")?;
        (module_key, shader)
    };

    let module = cache.get_or_create_module(module_name, shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(module_name, entry_point, &module, &layout);

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
///
/// Supports F32, I32, U32 dtypes.
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
    // Validate dtype support for this operation
    dtype_support::check_scalar_dtype_support(op, dtype)?;

    // Get entry point name based on dtype (cached, leaked once per op+dtype)
    let entry_point = get_or_leak_entry_point(op, dtype)?;

    // Use F32 shader for backward compatibility, or dtype-specific for I32/U32
    let (module_name, shader_source): (&str, &str) = if dtype == DType::F32 {
        ("elementwise", ELEMENTWISE_SHADER)
    } else {
        // For I32/U32, get cached shader and module key (leaked once per dtype+op_type)
        let shader = get_or_leak_shader(dtype, "scalar")?;
        let module_key = get_or_leak_module_key(dtype, "scalar")?;
        (module_key, shader)
    };

    let module = cache.get_or_create_module(module_name, shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(module_name, entry_point, &module, &layout);

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
///
/// Supports F32, I32, U32 dtypes. Output is always F32.
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
    // Validate dtype support for this operation
    dtype_support::check_compare_dtype_support(op, dtype)?;

    // Get entry point name based on dtype (cached, leaked once per op+dtype)
    let entry_point = get_or_leak_entry_point(op, dtype)?;

    // Use F32 shader for backward compatibility, or dtype-specific for I32/U32
    let (module_name, shader_source): (&str, &str) = if dtype == DType::F32 {
        ("elementwise", ELEMENTWISE_SHADER)
    } else {
        // For I32/U32, get cached shader and module key (leaked once per dtype+op_type)
        let shader = get_or_leak_shader(dtype, "compare")?;
        let module_key = get_or_leak_module_key(dtype, "compare")?;
        (module_key, shader)
    };

    let module = cache.get_or_create_module(module_name, shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(module_name, entry_point, &module, &layout);

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
// Cast Operation (uses generator for DRY)
// ============================================================================

/// Get static module name and entry point for a cast operation.
///
/// Returns (module_name, entry_point) for caching purposes.
/// The shader source is generated dynamically via `generate_cast_shader()`.
fn cast_info(src: DType, dst: DType) -> Option<(&'static str, &'static str)> {
    match (src, dst) {
        (DType::F32, DType::I32) => Some(("cast_f32_i32", "cast_f32_to_i32")),
        (DType::F32, DType::U32) => Some(("cast_f32_u32", "cast_f32_to_u32")),
        (DType::I32, DType::F32) => Some(("cast_i32_f32", "cast_i32_to_f32")),
        (DType::I32, DType::U32) => Some(("cast_i32_u32", "cast_i32_to_u32")),
        (DType::U32, DType::F32) => Some(("cast_u32_f32", "cast_u32_to_f32")),
        (DType::U32, DType::I32) => Some(("cast_u32_i32", "cast_u32_to_i32")),
        _ => None,
    }
}

/// Launch cast operation kernel.
///
/// Converts `out[i] = dst_dtype(a[i])` for all elements.
/// Supports F32 ↔ I32 ↔ U32 conversions.
///
/// Uses `generate_cast_shader()` from the generator module for DRY shader generation.
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
    // Same-type cast is a no-op (should be caught earlier, but handle here too)
    if src_dtype == dst_dtype {
        return Ok(());
    }

    // Get static names for caching
    let (module_name, entry_point) =
        cast_info(src_dtype, dst_dtype).ok_or_else(|| Error::UnsupportedDType {
            dtype: src_dtype,
            op: "cast (unsupported dtype combination)",
        })?;

    // Generate shader source dynamically (DRY - single source of truth in generator.rs)
    let shader_source = generate_cast_shader(src_dtype, dst_dtype)?;

    let module = cache.get_or_create_module(module_name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
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
