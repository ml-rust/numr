//! WGSL kernel launchers for special mathematical functions
//!
//! Provides native GPU implementations for erf, erfc, erfinv, gamma,
//! lgamma, digamma, beta, betainc, gammainc, gammaincc.

use wgpu::util::DeviceExt;
use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Static WGSL Shader Sources
// ============================================================================

const SPECIAL_UNARY_F32: &str = include_str!("special_unary_f32.wgsl");
const SPECIAL_BINARY_F32: &str = include_str!("special_binary_f32.wgsl");
const SPECIAL_TERNARY_F32: &str = include_str!("special_ternary_f32.wgsl");

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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "special_unary",
        });
    }
    let entry_point = format!("{}_f32", op);
    let module_key = "special_unary_f32";

    let module = pipeline_cache.get_or_create_module(module_key, SPECIAL_UNARY_F32);

    // Layout: 2 storage buffers (input, output) + 1 uniform (params)
    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);

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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "special_binary",
        });
    }
    let entry_point = format!("{}_f32", op);
    let module_key = "special_binary_f32";

    let module = pipeline_cache.get_or_create_module(module_key, SPECIAL_BINARY_F32);

    // Layout: 3 storage buffers (input_a, input_b, output) + 1 uniform (params)
    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);

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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "special_ternary",
        });
    }
    let entry_point = format!("{}_f32", op);
    let module_key = "special_ternary_f32";

    let module = pipeline_cache.get_or_create_module(module_key, SPECIAL_TERNARY_F32);

    // Layout: 4 storage buffers (input_a, input_b, input_x, output) + 1 uniform (params)
    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);

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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "special_unary_with_int",
        });
    }
    let entry_point = format!("{}_f32", op);
    let module_key = "special_unary_f32";

    let module = pipeline_cache.get_or_create_module(module_key, SPECIAL_UNARY_F32);

    // Layout: 2 storage buffers + 1 uniform (params with numel and n)
    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);

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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "special_unary_with_two_ints",
        });
    }
    let entry_point = format!("{}_f32", op);
    let module_key = "special_unary_f32";

    let module = pipeline_cache.get_or_create_module(module_key, SPECIAL_UNARY_F32);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);

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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "special_binary_with_two_ints",
        });
    }
    let entry_point = format!("{}_f32", op);
    let module_key = "special_binary_f32";

    let module = pipeline_cache.get_or_create_module(module_key, SPECIAL_BINARY_F32);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);

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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "special_unary_with_2f32",
        });
    }
    let entry_point = format!("{}_f32", op);
    let module_key = "special_unary_f32";

    let module = pipeline_cache.get_or_create_module(module_key, SPECIAL_UNARY_F32);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);

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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "special_unary_with_3f32",
        });
    }
    let entry_point = format!("{}_f32", op);
    let module_key = "special_unary_f32";

    let module = pipeline_cache.get_or_create_module(module_key, SPECIAL_UNARY_F32);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        pipeline_cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);

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
