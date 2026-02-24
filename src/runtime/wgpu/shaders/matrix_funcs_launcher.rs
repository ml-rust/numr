//! GPU launchers for matrix function operations on quasi-triangular matrices.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache};
use crate::dtype::DType;
use crate::error::{Error, Result};

const VALIDATE_EIGENVALUES_SHADER: &str = include_str!("validate_eigenvalues_f32.wgsl");
// entry point: "validate_eigenvalues_f32"

const DIAGONAL_EXP_SHADER: &str = include_str!("diagonal_exp_f32.wgsl");
// entry point: "diagonal_exp_f32"

const DIAGONAL_LOG_SHADER: &str = include_str!("diagonal_log_f32.wgsl");
// entry point: "diagonal_log_f32"

const DIAGONAL_SQRT_SHADER: &str = include_str!("diagonal_sqrt_f32.wgsl");
// entry point: "diagonal_sqrt_f32"

const PARLETT_COLUMN_SHADER: &str = include_str!("parlett_column_f32.wgsl");
// entry point: "parlett_column_f32"

fn check_dtype_f32(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Launch eigenvalue validation on Schur form.
///
/// Returns validation result in `result` buffer:
/// - `result[0]` = 1.0 if any non-positive real eigenvalue found, 0.0 otherwise
/// - `result[1]` = the problematic eigenvalue value (if any)
pub fn launch_validate_eigenvalues(
    cache: &PipelineCache,
    queue: &Queue,
    matrix_t: &Buffer,
    result: &Buffer,
    n: usize,
    eps: f32,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32(dtype, "validate_eigenvalues")?;

    let module =
        cache.get_or_create_module("validate_eigenvalues_f32", VALIDATE_EIGENVALUES_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "validate_eigenvalues_f32",
        "validate_eigenvalues_f32",
        &module,
        &layout,
    );

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
    check_dtype_f32(dtype, "diagonal_func")?;

    let (shader_src, module_name, entry_point): (&str, &'static str, &'static str) = match func_type
    {
        "exp" => (DIAGONAL_EXP_SHADER, "diagonal_exp_f32", "diagonal_exp_f32"),
        "log" => (DIAGONAL_LOG_SHADER, "diagonal_log_f32", "diagonal_log_f32"),
        "sqrt" => (
            DIAGONAL_SQRT_SHADER,
            "diagonal_sqrt_f32",
            "diagonal_sqrt_f32",
        ),
        _ => {
            return Err(Error::Internal(format!(
                "Unknown diagonal func type: {}",
                func_type
            )));
        }
    };

    let module = cache.get_or_create_module(module_name, shader_src);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_name, entry_point, &module, &layout);

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
    check_dtype_f32(dtype, "parlett_column")?;

    let module = cache.get_or_create_module("parlett_column_f32", PARLETT_COLUMN_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("parlett_column_f32", "parlett_column_f32", &module, &layout);

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
