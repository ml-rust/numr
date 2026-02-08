//! Matrix functions: exponential, square root, logarithm

use wgpu::{Buffer, Queue};

use super::check_dtype_f32;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::wgpu::shaders::linalg_shaders::matrix_functions::MATRIX_FUNCTIONS_SHADER;
use crate::runtime::wgpu::shaders::pipeline::{LayoutKey, PipelineCache};

/// Launch matrix exponential kernel for quasi-triangular matrix.
///
/// Computes exp(T) using Parlett recurrence where T is in quasi-triangular form.
///
/// # Arguments
/// * `t` - Input quasi-triangular matrix buffer [n * n]
/// * `result` - Output buffer for exp(T) [n * n]
/// * `params_buffer` - Parameters buffer with [n, max_iter]
pub fn launch_exp_quasi_triangular(
    cache: &PipelineCache,
    queue: &Queue,
    t: &Buffer,
    result: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "exp_quasi_triangular");

    let module = cache.get_or_create_module("linalg_matrix_functions", MATRIX_FUNCTIONS_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "linalg_matrix_functions",
        "exp_quasi_triangular_f32",
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(&layout, &[t, result, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("exp_quasi_triangular"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("exp_quasi_triangular"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch matrix square root kernel using Denman-Beavers iteration.
///
/// Computes sqrt(T) for input matrix T.
///
/// # Arguments
/// * `input` - Input matrix buffer [n * n]
/// * `y` - Y iteration buffer [n * n]
/// * `z` - Z iteration buffer [n * n]
/// * `work1` - Work buffer 1 [n * n]
/// * `work2` - Work buffer 2 [n * n]
/// * `params_buffer` - Parameters buffer with [n, max_iter]
pub fn launch_sqrt_quasi_triangular(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    y: &Buffer,
    z: &Buffer,
    work1: &Buffer,
    work2: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "sqrt_quasi_triangular");

    let module = cache.get_or_create_module("linalg_matrix_functions", MATRIX_FUNCTIONS_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "linalg_matrix_functions",
        "sqrt_quasi_triangular_f32",
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(&layout, &[input, y, z, work1, work2, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sqrt_quasi_triangular"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sqrt_quasi_triangular"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch matrix logarithm kernel using inverse scaling and squaring.
///
/// Computes log(T) for input matrix T.
///
/// # Arguments
/// * `input` - Input matrix buffer [n * n]
/// * `work` - Work buffer [n * n]
/// * `result` - Output buffer for log(T) [n * n]
/// * `temp` - Temp buffer [n * n]
/// * `xpower` - X power buffer [n * n]
/// * `params_buffer` - Parameters buffer with [n, max_iter]
pub fn launch_log_quasi_triangular(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    work: &Buffer,
    result: &Buffer,
    temp: &Buffer,
    xpower: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "log_quasi_triangular");

    let module = cache.get_or_create_module("linalg_matrix_functions", MATRIX_FUNCTIONS_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "linalg_matrix_functions",
        "log_quasi_triangular_f32",
        &module,
        &layout,
    );

    let bind_group =
        cache.create_bind_group(&layout, &[input, work, result, temp, xpower, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("log_quasi_triangular"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("log_quasi_triangular"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
