//! WGSL kernel launchers for special mathematical functions
//!
//! Provides native GPU implementations for erf, erfc, erfinv, gamma,
//! lgamma, digamma, beta, betainc, gammainc, gammaincc.

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

use wgpu::util::DeviceExt;
use wgpu::{Buffer, Queue};

use super::generator::{
    dtype_suffix, generate_special_binary_shader, generate_special_ternary_shader,
    generate_special_unary_shader,
};
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::Result;

// ============================================================================
// Shader Module Cache
// ============================================================================

static SPECIAL_UNARY_CACHE: OnceLock<RwLock<HashMap<DType, &'static str>>> = OnceLock::new();
static SPECIAL_BINARY_CACHE: OnceLock<RwLock<HashMap<DType, &'static str>>> = OnceLock::new();
static SPECIAL_TERNARY_CACHE: OnceLock<RwLock<HashMap<DType, &'static str>>> = OnceLock::new();

fn get_or_leak_special_unary_shader(dtype: DType) -> Result<&'static str> {
    let cache = SPECIAL_UNARY_CACHE.get_or_init(|| RwLock::new(HashMap::new()));

    {
        let read_guard = read_lock(cache);
        if let Some(&shader_ref) = read_guard.get(&dtype) {
            return Ok(shader_ref);
        }
    }

    let shader = generate_special_unary_shader(dtype)?;
    let leaked: &'static str = Box::leak(shader.into_boxed_str());

    let mut write_guard = write_lock(cache);
    write_guard.insert(dtype, leaked);

    Ok(leaked)
}

fn get_or_leak_special_binary_shader(dtype: DType) -> Result<&'static str> {
    let cache = SPECIAL_BINARY_CACHE.get_or_init(|| RwLock::new(HashMap::new()));

    {
        let read_guard = read_lock(cache);
        if let Some(&shader_ref) = read_guard.get(&dtype) {
            return Ok(shader_ref);
        }
    }

    let shader = generate_special_binary_shader(dtype)?;
    let leaked: &'static str = Box::leak(shader.into_boxed_str());

    let mut write_guard = write_lock(cache);
    write_guard.insert(dtype, leaked);

    Ok(leaked)
}

fn get_or_leak_special_ternary_shader(dtype: DType) -> Result<&'static str> {
    let cache = SPECIAL_TERNARY_CACHE.get_or_init(|| RwLock::new(HashMap::new()));

    {
        let read_guard = read_lock(cache);
        if let Some(&shader_ref) = read_guard.get(&dtype) {
            return Ok(shader_ref);
        }
    }

    let shader = generate_special_ternary_shader(dtype)?;
    let leaked: &'static str = Box::leak(shader.into_boxed_str());

    let mut write_guard = write_lock(cache);
    write_guard.insert(dtype, leaked);

    Ok(leaked)
}

// ============================================================================
// Unary Special Functions (erf, erfc, erfinv, gamma, lgamma, digamma)
// ============================================================================

/// Launch a special unary function kernel
pub fn launch_special_unary(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    op: &str,
    input: &Buffer,
    output: &Buffer,
    numel: u32,
    dtype: DType,
) -> Result<()> {
    let shader = get_or_leak_special_unary_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("{}_{}", op, suffix);
    let module_key = format!("special_unary_{}", suffix);

    let module = pipeline_cache.get_or_create_module_from_source(&module_key, shader);

    // Layout: 2 storage buffers (input, output) + 1 uniform (params)
    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(&module_key, &entry_point, &module, &layout);

    // Create params buffer
    let params_data = [numel];
    let params_buffer =
        pipeline_cache
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("special_unary_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

    // Create bind group
    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, &params_buffer]);

    // Dispatch
    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("special_unary_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("special_unary_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel as usize), 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    Ok(())
}

// ============================================================================
// Binary Special Functions (beta, gammainc, gammaincc)
// ============================================================================

/// Launch a special binary function kernel
pub fn launch_special_binary(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    op: &str,
    input_a: &Buffer,
    input_b: &Buffer,
    output: &Buffer,
    numel: u32,
    dtype: DType,
) -> Result<()> {
    let shader = get_or_leak_special_binary_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("{}_{}", op, suffix);
    let module_key = format!("special_binary_{}", suffix);

    let module = pipeline_cache.get_or_create_module_from_source(&module_key, shader);

    // Layout: 3 storage buffers (input_a, input_b, output) + 1 uniform (params)
    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(&module_key, &entry_point, &module, &layout);

    // Create params buffer
    let params_data = [numel];
    let params_buffer =
        pipeline_cache
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("special_binary_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

    // Create bind group
    let bind_group =
        pipeline_cache.create_bind_group(&layout, &[input_a, input_b, output, &params_buffer]);

    // Dispatch
    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("special_binary_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("special_binary_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel as usize), 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    Ok(())
}

// ============================================================================
// Ternary Special Functions (betainc)
// ============================================================================

/// Launch a special ternary function kernel (betainc)
pub fn launch_special_ternary(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    op: &str,
    input_a: &Buffer,
    input_b: &Buffer,
    input_x: &Buffer,
    output: &Buffer,
    numel: u32,
    dtype: DType,
) -> Result<()> {
    let shader = get_or_leak_special_ternary_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("{}_{}", op, suffix);
    let module_key = format!("special_ternary_{}", suffix);

    let module = pipeline_cache.get_or_create_module_from_source(&module_key, shader);

    // Layout: 4 storage buffers (input_a, input_b, input_x, output) + 1 uniform (params)
    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(&module_key, &entry_point, &module, &layout);

    // Create params buffer
    let params_data = [numel];
    let params_buffer =
        pipeline_cache
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("special_ternary_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

    // Create bind group
    let bind_group = pipeline_cache.create_bind_group(
        &layout,
        &[input_a, input_b, input_x, output, &params_buffer],
    );

    // Dispatch
    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("special_ternary_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("special_ternary_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel as usize), 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    Ok(())
}

// ============================================================================
// Extended Special Functions with Parameters
// ============================================================================

/// Launch a special unary function kernel with one i32 parameter (legendre_p)
pub fn launch_special_unary_with_int(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    op: &str,
    input: &Buffer,
    output: &Buffer,
    numel: u32,
    n: i32,
    dtype: DType,
) -> Result<()> {
    let shader = get_or_leak_special_unary_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("{}_{}", op, suffix);
    let module_key = format!("special_unary_{}", suffix);

    let module = pipeline_cache.get_or_create_module_from_source(&module_key, shader);

    // Layout: 2 storage buffers + 1 uniform (params with numel and n)
    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(&module_key, &entry_point, &module, &layout);

    // Create params buffer with numel and n
    let params_data = [numel, n as u32];
    let params_buffer =
        pipeline_cache
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("special_unary_int_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, &params_buffer]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("special_unary_int_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("special_unary_int_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel as usize), 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    Ok(())
}

/// Launch a special unary function kernel with two i32 parameters (legendre_p_assoc)
pub fn launch_special_unary_with_two_ints(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    op: &str,
    input: &Buffer,
    output: &Buffer,
    numel: u32,
    n: i32,
    m: i32,
    dtype: DType,
) -> Result<()> {
    let shader = get_or_leak_special_unary_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("{}_{}", op, suffix);
    let module_key = format!("special_unary_{}", suffix);

    let module = pipeline_cache.get_or_create_module_from_source(&module_key, shader);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(&module_key, &entry_point, &module, &layout);

    // Create params buffer with numel, n, m
    let params_data = [numel, n as u32, m as u32, 0u32]; // Pad to 16 bytes
    let params_buffer =
        pipeline_cache
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("special_unary_two_ints_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, &params_buffer]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("special_unary_two_ints_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("special_unary_two_ints_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel as usize), 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    Ok(())
}

/// Launch a special binary function kernel with two i32 parameters (sph_harm)
pub fn launch_special_binary_with_two_ints(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    op: &str,
    input_a: &Buffer,
    input_b: &Buffer,
    output: &Buffer,
    numel: u32,
    n: i32,
    m: i32,
    dtype: DType,
) -> Result<()> {
    let shader = get_or_leak_special_binary_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("{}_{}", op, suffix);
    let module_key = format!("special_binary_{}", suffix);

    let module = pipeline_cache.get_or_create_module_from_source(&module_key, shader);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(&module_key, &entry_point, &module, &layout);

    // Create params buffer with numel, n, m
    let params_data = [numel, n as u32, m as u32, 0u32]; // Pad to 16 bytes
    let params_buffer =
        pipeline_cache
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("special_binary_two_ints_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

    let bind_group =
        pipeline_cache.create_bind_group(&layout, &[input_a, input_b, output, &params_buffer]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("special_binary_two_ints_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("special_binary_two_ints_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel as usize), 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    Ok(())
}

/// Launch a special unary function kernel with two f32 parameters (hyp1f1)
pub fn launch_special_unary_with_2f32(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    op: &str,
    input: &Buffer,
    output: &Buffer,
    numel: u32,
    a: f32,
    b: f32,
    dtype: DType,
) -> Result<()> {
    let shader = get_or_leak_special_unary_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("{}_{}", op, suffix);
    let module_key = format!("special_unary_{}", suffix);

    let module = pipeline_cache.get_or_create_module_from_source(&module_key, shader);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(&module_key, &entry_point, &module, &layout);

    // Create params buffer with numel, a, b (use u32 + 2 f32s)
    let numel_bits = numel;
    let params_data: [u32; 4] = [numel_bits, 0, a.to_bits(), b.to_bits()];
    let params_buffer =
        pipeline_cache
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("special_unary_2f32_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, &params_buffer]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("special_unary_2f32_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("special_unary_2f32_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel as usize), 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    Ok(())
}

/// Launch a special unary function kernel with three f32 parameters (hyp2f1)
pub fn launch_special_unary_with_3f32(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    op: &str,
    input: &Buffer,
    output: &Buffer,
    numel: u32,
    a: f32,
    b: f32,
    c: f32,
    dtype: DType,
) -> Result<()> {
    let shader = get_or_leak_special_unary_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("{}_{}", op, suffix);
    let module_key = format!("special_unary_{}", suffix);

    let module = pipeline_cache.get_or_create_module_from_source(&module_key, shader);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(&module_key, &entry_point, &module, &layout);

    // Create params buffer with numel, a, b, c
    let params_data: [u32; 6] = [numel, 0, a.to_bits(), b.to_bits(), c.to_bits(), 0];
    let params_buffer =
        pipeline_cache
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("special_unary_3f32_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, &params_buffer]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("special_unary_3f32_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("special_unary_3f32_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel as usize), 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    Ok(())
}
