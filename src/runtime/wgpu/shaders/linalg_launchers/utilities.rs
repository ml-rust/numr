//! Utility operations: determinant, permutation, column ops, rank helpers

use wgpu::{Buffer, Queue};

use super::check_dtype_f32;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::wgpu::shaders::linalg_shaders::utilities::UTILITIES_SHADER;
use crate::runtime::wgpu::shaders::pipeline::{LayoutKey, PipelineCache, workgroup_count};

/// Launch determinant computation from LU decomposition.
pub fn launch_det_from_lu(
    cache: &PipelineCache,
    queue: &Queue,
    lu_matrix: &Buffer,
    det_output: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "det_from_lu");

    let module = cache.get_or_create_module("linalg_utilities", UTILITIES_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg_utilities", "det_from_lu_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[lu_matrix, det_output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("det_from_lu"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("det_from_lu"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Apply LU permutation to a vector.
pub fn launch_apply_lu_permutation(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    pivots: &Buffer,
    params_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "apply_lu_permutation");

    let module = cache.get_or_create_module("linalg_utilities", UTILITIES_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg", "apply_lu_permutation_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, output, pivots, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("apply_lu_permutation"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("apply_lu_permutation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Scatter vector into a column of a matrix.
pub fn launch_scatter_column(
    cache: &PipelineCache,
    queue: &Queue,
    vec: &Buffer,
    matrix: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "scatter_column");

    let module = cache.get_or_create_module("linalg_utilities", UTILITIES_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg_utilities", "scatter_column_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[vec, matrix, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scatter_column"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter_column"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Extract a column from a matrix.
pub fn launch_extract_column(
    cache: &PipelineCache,
    queue: &Queue,
    matrix: &Buffer,
    col_out: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "extract_column");

    let module = cache.get_or_create_module("linalg_utilities", UTILITIES_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg_utilities", "extract_column_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[matrix, col_out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("extract_column"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("extract_column"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(m), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Compute maximum absolute value of elements.
pub fn launch_max_abs(
    cache: &PipelineCache,
    queue: &Queue,
    values: &Buffer,
    max_output: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "max_abs");

    let module = cache.get_or_create_module("linalg_utilities", UTILITIES_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg_utilities", "max_abs_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[values, max_output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("max_abs"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("max_abs"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Count elements with absolute value above threshold.
pub fn launch_count_above_threshold(
    cache: &PipelineCache,
    queue: &Queue,
    values: &Buffer,
    count_output: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "count_above_threshold");

    let module = cache.get_or_create_module("linalg_utilities", UTILITIES_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg", "count_above_threshold_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[values, count_output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("count_above_threshold"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("count_above_threshold"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Copy matrix data on device.
pub fn launch_matrix_copy(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "matrix_copy");

    let module = cache.get_or_create_module("linalg_utilities", UTILITIES_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("linalg_utilities", "matrix_copy_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, dst, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matrix_copy"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matrix_copy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
