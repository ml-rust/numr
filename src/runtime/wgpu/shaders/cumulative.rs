//! Cumulative operation WGSL kernel launchers
//!
//! Provides launchers for cumulative operations:
//! - `cumsum` - Cumulative sum along a dimension
//! - `cumprod` - Cumulative product along a dimension
//! - `logsumexp` - Numerically stable log-sum-exp reduction

use wgpu::{Buffer, Queue};

use super::generator::{
    dtype_suffix, generate_cumprod_shader, generate_cumprod_strided_shader, generate_cumsum_shader,
    generate_cumsum_strided_shader, generate_logsumexp_shader, generate_logsumexp_strided_shader,
};
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::Result;

// ============================================================================
// Cumulative Sum
// ============================================================================

/// Launch cumsum operation kernel (contiguous data).
///
/// Parameters:
/// - scan_size: Size of the dimension being scanned
/// - outer_size: Number of independent scans
pub fn launch_cumsum(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    outer_size: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point_name = format!("cumsum_{}", suffix);

    // Generate shader on-demand
    let shader_source = generate_cumsum_shader(dtype)?;

    let module_name = format!("cumsum_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline =
        cache.get_or_create_dynamic_pipeline("cumsum", &entry_point_name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cumsum"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cumsum"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(outer_size), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch strided cumsum operation kernel.
pub fn launch_cumsum_strided(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_inner: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point_name = format!("cumsum_strided_{}", suffix);

    let shader_source = generate_cumsum_strided_shader(dtype)?;

    let module = cache
        .get_or_create_module_from_source(&format!("cumsum_strided_{}", suffix), &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline =
        cache.get_or_create_dynamic_pipeline("cumsum_strided", &entry_point_name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cumsum_strided"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cumsum_strided"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_inner), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Cumulative Product
// ============================================================================

/// Launch cumprod operation kernel (contiguous data).
pub fn launch_cumprod(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    outer_size: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point_name = format!("cumprod_{}", suffix);

    let shader_source = generate_cumprod_shader(dtype)?;

    let module =
        cache.get_or_create_module_from_source(&format!("cumprod_{}", suffix), &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline =
        cache.get_or_create_dynamic_pipeline("cumprod", &entry_point_name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cumprod"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cumprod"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(outer_size), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch strided cumprod operation kernel.
pub fn launch_cumprod_strided(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_inner: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point_name = format!("cumprod_strided_{}", suffix);

    let shader_source = generate_cumprod_strided_shader(dtype)?;

    let module = cache
        .get_or_create_module_from_source(&format!("cumprod_strided_{}", suffix), &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_dynamic_pipeline(
        "cumprod_strided",
        &entry_point_name,
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cumprod_strided"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cumprod_strided"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_inner), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Log-Sum-Exp
// ============================================================================

/// Launch logsumexp operation kernel (contiguous data).
pub fn launch_logsumexp(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    outer_size: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point_name = format!("logsumexp_{}", suffix);

    let shader_source = generate_logsumexp_shader(dtype)?;

    let module =
        cache.get_or_create_module_from_source(&format!("logsumexp_{}", suffix), &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline =
        cache.get_or_create_dynamic_pipeline("logsumexp", &entry_point_name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("logsumexp"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("logsumexp"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(outer_size), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch strided logsumexp operation kernel.
pub fn launch_logsumexp_strided(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_inner: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point_name = format!("logsumexp_strided_{}", suffix);

    let shader_source = generate_logsumexp_strided_shader(dtype)?;

    let module = cache
        .get_or_create_module_from_source(&format!("logsumexp_strided_{}", suffix), &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_dynamic_pipeline(
        "logsumexp_strided",
        &entry_point_name,
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("logsumexp_strided"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("logsumexp_strided"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_inner), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
