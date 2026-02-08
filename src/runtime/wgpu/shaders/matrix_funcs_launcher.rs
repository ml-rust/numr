//! GPU launchers for matrix function operations on quasi-triangular matrices.

use wgpu::{Buffer, Queue};

use super::generator::{
    dtype_suffix, generate_diagonal_func_shader, generate_parlett_column_shader,
    generate_validate_eigenvalues_shader,
};
use super::pipeline::{LayoutKey, PipelineCache};
use crate::dtype::DType;
use crate::error::Result;

/// Launch eigenvalue validation on Schur form.
///
/// Returns validation result in `result` buffer:
/// - result[0] = 1.0 if any non-positive real eigenvalue found, 0.0 otherwise
/// - result[1] = the problematic eigenvalue value (if any)
pub fn launch_validate_eigenvalues(
    cache: &PipelineCache,
    queue: &Queue,
    matrix_t: &Buffer,
    result: &Buffer,
    n: usize,
    eps: f32,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let shader_key = format!("validate_eigenvalues_{}", suffix);
    let entry_point = format!("validate_eigenvalues_{}", suffix);

    let shader_source = generate_validate_eigenvalues_shader(dtype)?;
    let module = cache.get_or_create_module_from_source(&shader_key, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_dynamic_pipeline(&shader_key, &entry_point, &module, &layout);

    // Create params buffer
    let params: [u32; 4] = [n as u32, eps.to_bits(), 0, 0];
    let params_buffer = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("validate_eigenvalues_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

    let bind_group = cache.create_bind_group(&layout, &[matrix_t, result, &params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("validate_eigenvalues"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("validate_eigenvalues"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single workgroup - sequential algorithm
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch diagonal function application (exp, log, sqrt) on Schur form.
///
/// Applies the function to 1x1 and 2x2 diagonal blocks.
pub fn launch_diagonal_func(
    cache: &PipelineCache,
    queue: &Queue,
    input_t: &Buffer,
    output_f: &Buffer,
    n: usize,
    eps: f32,
    func_type: &str,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let shader_key = format!("diagonal_{}_{}", func_type, suffix);
    let entry_point = format!("diagonal_{}_{}", func_type, suffix);

    let shader_source = generate_diagonal_func_shader(dtype, func_type)?;
    let module = cache.get_or_create_module_from_source(&shader_key, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_dynamic_pipeline(&shader_key, &entry_point, &module, &layout);

    // Create params buffer
    let params: [u32; 4] = [n as u32, eps.to_bits(), 0, 0];
    let params_buffer = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("diagonal_func_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

    let bind_group = cache.create_bind_group(&layout, &[input_t, output_f, &params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("diagonal_func"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("diagonal_func"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single workgroup - sequential block processing
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch Parlett recurrence for a single column of off-diagonal elements.
///
/// Must be called for columns 1..n in order (Parlett's algorithm is column-wise sequential).
pub fn launch_parlett_column(
    cache: &PipelineCache,
    queue: &Queue,
    input_t: &Buffer,
    output_f: &Buffer,
    n: usize,
    col: usize,
    eps: f32,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let shader_key = format!("parlett_column_{}", suffix);
    let entry_point = format!("parlett_column_{}", suffix);

    let shader_source = generate_parlett_column_shader(dtype)?;
    let module = cache.get_or_create_module_from_source(&shader_key, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_dynamic_pipeline(&shader_key, &entry_point, &module, &layout);

    // Create params buffer
    let params: [u32; 4] = [n as u32, col as u32, eps.to_bits(), 0];
    let params_buffer = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("parlett_column_params"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

    let bind_group = cache.create_bind_group(&layout, &[input_t, output_f, &params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("parlett_column"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("parlett_column"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // Each row i < col can be processed in parallel
        let workgroups = (col as u32 + 255) / 256;
        pass.dispatch_workgroups(workgroups.max(1), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Compute f(T) for quasi-triangular matrix T using GPU kernels.
///
/// This is the main entry point that:
/// 1. Applies function to diagonal blocks (GPU)
/// 2. Computes off-diagonal elements using Parlett's recurrence (GPU, column-by-column)
pub fn compute_schur_func_gpu(
    cache: &PipelineCache,
    queue: &Queue,
    input_t: &Buffer,
    output_f: &Buffer,
    n: usize,
    func_type: &str,
    dtype: DType,
) -> Result<()> {
    let eps = f32::EPSILON;

    // Step 1: Apply function to diagonal blocks
    launch_diagonal_func(cache, queue, input_t, output_f, n, eps, func_type, dtype)?;

    // Step 2: Compute off-diagonal elements column by column
    // Parlett's algorithm requires column-wise processing: for each column j,
    // we need F[i,k] and F[k,j] for all k between i and j
    for col in 1..n {
        launch_parlett_column(cache, queue, input_t, output_f, n, col, eps, dtype)?;
    }

    Ok(())
}
