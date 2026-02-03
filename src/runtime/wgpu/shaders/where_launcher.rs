//! Where (conditional select) WGSL kernel launchers
//!
//! Provides launchers for where_cond operations with multi-dtype support:
//! - `launch_where_op` - Legacy F32-only version for backward compatibility
//! - `launch_where_generic_op` - Generic condition dtype support (F32, I32, U32)
//! - `launch_where_broadcast_op` - Broadcast support with generic condition dtype

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

// ============================================================================
// Lock Helpers (Handle Poisoned Locks Gracefully)
// ============================================================================

/// Acquire read lock, recovering from poison if necessary.
fn read_lock<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Acquire write lock, recovering from poison if necessary.
fn write_lock<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

use wgpu::{Buffer, Queue};

use super::generator::{dtype_suffix, generate_where_cond_shader};
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::Result;

// ============================================================================
// Shader Caching
// ============================================================================

/// Cache for where_cond shader references (leaked once per cond_dtype+out_dtype combination)
static WHERE_SHADER_CACHE: OnceLock<RwLock<HashMap<(DType, DType), &'static str>>> =
    OnceLock::new();

/// Cache for where_cond module key references
static WHERE_MODULE_KEY_CACHE: OnceLock<RwLock<HashMap<(DType, DType), &'static str>>> =
    OnceLock::new();

/// Cache for where_cond entry point references
static WHERE_ENTRY_CACHE: OnceLock<RwLock<HashMap<(DType, DType, bool), &'static str>>> =
    OnceLock::new();

/// Get or generate where_cond shader for specific cond_dtype and out_dtype.
fn get_or_leak_where_shader(cond_dtype: DType, out_dtype: DType) -> Result<&'static str> {
    let cache = WHERE_SHADER_CACHE.get_or_init(|| RwLock::new(HashMap::new()));

    {
        let read_guard = read_lock(cache);
        if let Some(&shader_ref) = read_guard.get(&(cond_dtype, out_dtype)) {
            return Ok(shader_ref);
        }
    }

    let shader = generate_where_cond_shader(cond_dtype, out_dtype)?;
    let leaked: &'static str = Box::leak(shader.into_boxed_str());

    let mut write_guard = write_lock(cache);
    write_guard.insert((cond_dtype, out_dtype), leaked);

    Ok(leaked)
}

/// Get module key for where_cond shader.
fn get_or_leak_where_module_key(cond_dtype: DType, out_dtype: DType) -> Result<&'static str> {
    let cache = WHERE_MODULE_KEY_CACHE.get_or_init(|| RwLock::new(HashMap::new()));

    {
        let read_guard = read_lock(cache);
        if let Some(&key_ref) = read_guard.get(&(cond_dtype, out_dtype)) {
            return Ok(key_ref);
        }
    }

    let cond_suffix = dtype_suffix(cond_dtype)?;
    let out_suffix = dtype_suffix(out_dtype)?;
    let key = format!("where_cond_{}_{}", cond_suffix, out_suffix);
    let leaked: &'static str = Box::leak(key.into_boxed_str());

    let mut write_guard = write_lock(cache);
    write_guard.insert((cond_dtype, out_dtype), leaked);

    Ok(leaked)
}

/// Get entry point name for where_cond operation.
fn get_or_leak_where_entry(
    cond_dtype: DType,
    out_dtype: DType,
    broadcast: bool,
) -> Result<&'static str> {
    let cache = WHERE_ENTRY_CACHE.get_or_init(|| RwLock::new(HashMap::new()));

    {
        let read_guard = read_lock(cache);
        if let Some(&entry_ref) = read_guard.get(&(cond_dtype, out_dtype, broadcast)) {
            return Ok(entry_ref);
        }
    }

    let cond_suffix = dtype_suffix(cond_dtype)?;
    let out_suffix = dtype_suffix(out_dtype)?;
    let prefix = if broadcast {
        "where_broadcast_cond"
    } else {
        "where_cond"
    };
    let entry = format!("{}_{}_{}", prefix, cond_suffix, out_suffix);
    let leaked: &'static str = Box::leak(entry.into_boxed_str());

    let mut write_guard = write_lock(cache);
    write_guard.insert((cond_dtype, out_dtype, broadcast), leaked);

    Ok(leaked)
}

// ============================================================================
// Kernel Launchers
// ============================================================================

/// Launch where conditional operation kernel.
///
/// Computes `out[i] = cond[i] ? x[i] : y[i]` for all elements.
/// This is the legacy F32-only version for backward compatibility.
#[allow(clippy::too_many_arguments)]
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
    // Delegate to generic version with F32 condition
    launch_where_generic_op(
        cache,
        queue,
        cond,
        x,
        y,
        out,
        params_buffer,
        numel,
        DType::F32,
        dtype,
    )
}

/// Launch where conditional operation kernel with generic condition dtype.
///
/// Computes `out[i] = cond[i] != 0 ? x[i] : y[i]` for all elements.
/// Supports F32, I32, U32 condition dtypes.
#[allow(clippy::too_many_arguments)]
pub fn launch_where_generic_op(
    cache: &PipelineCache,
    queue: &Queue,
    cond: &Buffer,
    x: &Buffer,
    y: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    cond_dtype: DType,
    out_dtype: DType,
) -> Result<()> {
    let shader = get_or_leak_where_shader(cond_dtype, out_dtype)?;
    let module_key = get_or_leak_where_module_key(cond_dtype, out_dtype)?;
    let entry_point = get_or_leak_where_entry(cond_dtype, out_dtype, false)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[cond, x, y, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("where_cond"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("where_cond"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch broadcast where conditional operation kernel.
///
/// Computes `out[i] = cond[cond_offset] != 0 ? x[x_offset] : y[y_offset]`
/// with broadcasting support.
#[allow(clippy::too_many_arguments)]
pub fn launch_where_broadcast_op(
    cache: &PipelineCache,
    queue: &Queue,
    cond: &Buffer,
    x: &Buffer,
    y: &Buffer,
    out: &Buffer,
    cond_strides: &Buffer,
    x_strides: &Buffer,
    y_strides: &Buffer,
    out_shape: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    cond_dtype: DType,
    out_dtype: DType,
) -> Result<()> {
    let shader = get_or_leak_where_shader(cond_dtype, out_dtype)?;
    let module_key = get_or_leak_where_module_key(cond_dtype, out_dtype)?;
    let entry_point = get_or_leak_where_entry(cond_dtype, out_dtype, true)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 8,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            cond,
            x,
            y,
            out,
            cond_strides,
            x_strides,
            y_strides,
            out_shape,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("where_broadcast"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("where_broadcast"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
