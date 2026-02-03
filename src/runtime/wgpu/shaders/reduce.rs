//! Reduction WGSL kernel launchers
//!
//! Provides launchers for reduction operations including:
//! - Sum, Mean, Max, Min, Prod, Any, All reductions along specified dimensions
//! - Argmax, Argmin (returns indices)
//! - Softmax (numerically stable)
//!
//! Multi-dtype support: F32, I32, U32 (F16 requires shader-f16 extension)
//! All operations run entirely on GPU with no CPU fallback.

use std::collections::HashMap;
use std::sync::RwLock;

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use super::reduce_wgsl::{
    REDUCE_SHADER, generate_reduce_shader, is_float_only_op, is_supported_dtype,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Shader Module Cache
// ============================================================================

/// Cache for dtype-specific shader modules
/// Key: (dtype suffix), Value: generated shader source
static SHADER_CACHE: RwLock<Option<HashMap<DType, String>>> = RwLock::new(None);

/// Get or generate shader for a specific dtype
fn get_shader_for_dtype(dtype: DType) -> String {
    // Check cache first
    {
        let cache = SHADER_CACHE.read().unwrap();
        if let Some(ref map) = *cache
            && let Some(shader) = map.get(&dtype)
        {
            return shader.clone();
        }
    }

    // Generate and cache
    let shader = generate_reduce_shader(dtype);
    {
        let mut cache = SHADER_CACHE.write().unwrap();
        let map = cache.get_or_insert_with(HashMap::new);
        map.insert(dtype, shader.clone());
    }
    shader
}

/// Get the module key for a dtype
fn module_key(dtype: DType) -> String {
    match dtype {
        DType::F32 => "reduce_f32".to_string(),
        DType::I32 => "reduce_i32".to_string(),
        DType::U32 => "reduce_u32".to_string(),
        _ => "reduce_f32".to_string(), // Fallback
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if dtype is supported, returning appropriate error if not
fn check_dtype_supported(dtype: DType, op: &'static str) -> Result<()> {
    if !is_supported_dtype(dtype) {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    // Float-only operations (mean, softmax) require F32
    if is_float_only_op(op) && dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

/// Get entry point name for reduce operation
fn reduce_entry_point(op: &str, dtype: DType) -> String {
    let suffix = match dtype {
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::U32 => "u32",
        _ => "f32",
    };
    format!("reduce_{}_{}", op, suffix)
}

/// Get entry point name for full reduce operation
fn full_reduce_entry_point(op: &str, dtype: DType) -> String {
    let suffix = match dtype {
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::U32 => "u32",
        _ => "f32",
    };
    format!("full_reduce_{}_{}", op, suffix)
}

/// Get entry point name for argreduce operation
fn argreduce_entry_point(op: &str, dtype: DType) -> String {
    let suffix = match dtype {
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::U32 => "u32",
        _ => "f32",
    };
    format!("{}_{}", op, suffix)
}

// ============================================================================
// Single-Dimension Reduction
// ============================================================================

/// Launch a reduction operation kernel along a single dimension.
///
/// Supports F32, I32, U32 dtypes. Mean is F32-only.
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
    check_dtype_supported(dtype, op)?;

    let entry_point = reduce_entry_point(op, dtype);
    // Leak entry_point to get static reference (cached, so leak is acceptable)
    let static_entry_point: &'static str = Box::leak(entry_point.into_boxed_str());

    // Use F32 shader for backward compatibility, or dtype-specific for I32/U32
    let (module_name, shader_source): (&str, &str) = if dtype == DType::F32 {
        ("reduce", REDUCE_SHADER)
    } else {
        // For I32/U32, we need to use the generated shader
        // But since we can't easily pass owned String to get_or_create_module,
        // we'll use a static approach with leaked strings (acceptable for caching)
        let shader = get_shader_for_dtype(dtype);
        let key = module_key(dtype);
        // Leak the strings to get static references (these are cached, so leak is acceptable)
        let static_key: &'static str = Box::leak(key.into_boxed_str());
        let static_shader: &'static str = Box::leak(shader.into_boxed_str());
        (static_key, static_shader)
    };

    let module = cache.get_or_create_module(module_name, shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(module_name, static_entry_point, &module, &layout);

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
/// Supports F32, I32, U32 dtypes.
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
    check_dtype_supported(dtype, op)?;

    let entry_point = full_reduce_entry_point(op, dtype);
    // Leak entry_point to get static reference (cached, so leak is acceptable)
    let static_entry_point: &'static str = Box::leak(entry_point.into_boxed_str());

    let (module_name, shader_source): (&str, &str) = if dtype == DType::F32 {
        ("reduce", REDUCE_SHADER)
    } else {
        let shader = get_shader_for_dtype(dtype);
        let key = module_key(dtype);
        let static_key: &'static str = Box::leak(key.into_boxed_str());
        let static_shader: &'static str = Box::leak(shader.into_boxed_str());
        (static_key, static_shader)
    };

    let module = cache.get_or_create_module(module_name, shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(module_name, static_entry_point, &module, &layout);

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
/// Supports F32, I32, U32 dtypes.
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
    check_dtype_supported(dtype, op)?;

    let entry_point = argreduce_entry_point(op, dtype);
    // Leak entry_point to get static reference (cached, so leak is acceptable)
    let static_entry_point: &'static str = Box::leak(entry_point.into_boxed_str());

    let (module_name, shader_source): (&str, &str) = if dtype == DType::F32 {
        ("reduce", REDUCE_SHADER)
    } else {
        let shader = get_shader_for_dtype(dtype);
        let key = module_key(dtype);
        let static_key: &'static str = Box::leak(key.into_boxed_str());
        let static_shader: &'static str = Box::leak(shader.into_boxed_str());
        (static_key, static_shader)
    };

    let module = cache.get_or_create_module(module_name, shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(module_name, static_entry_point, &module, &layout);

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
/// F32 only - softmax is a floating-point operation.
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
    check_dtype_supported(dtype, "softmax")?;

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
