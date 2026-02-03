//! Sparse linear algebra WGSL kernel launchers
//!
//! Provides launchers for sparse matrix filtering operations:
//! - `split_lu_count` - Count L and U non-zeros per row
//! - `split_lu_scatter` - Scatter values into L and U matrices
//! - `extract_lower_count` - Count lower triangle non-zeros per row
//! - `extract_lower_scatter` - Scatter lower triangle values

use wgpu::{Buffer, Queue};

use super::generator::dtype_suffix;
use super::generator::sparse_linalg::{
    generate_extract_lower_count_shader, generate_extract_lower_scatter_shader,
    generate_split_lu_count_shader, generate_split_lu_scatter_l_shader,
    generate_split_lu_scatter_u_shader,
};
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::Result;

// ============================================================================
// Split LU Operations
// ============================================================================

/// Launch split_lu count kernel - counts L and U non-zeros per row.
///
/// Buffers:
/// - row_ptrs: Input CSR row pointers (I32)
/// - col_indices: Input CSR column indices (I32)
/// - l_counts: Output L counts per row (I32)
/// - u_counts: Output U counts per row (I32)
/// - params: Uniform buffer with n (matrix size)
pub fn launch_split_lu_count(
    cache: &PipelineCache,
    queue: &Queue,
    row_ptrs: &Buffer,
    col_indices: &Buffer,
    l_counts: &Buffer,
    u_counts: &Buffer,
    params_buffer: &Buffer,
    n: usize,
) -> Result<()> {
    let shader_source = generate_split_lu_count_shader();
    let module = cache.get_or_create_module_from_source("split_lu_count", &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });
    let pipeline =
        cache.get_or_create_pipeline("split_lu_count", "split_lu_count", &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[row_ptrs, col_indices, l_counts, u_counts, params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("split_lu_count"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("split_lu_count"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch split_lu scatter L kernel - scatters values into L matrix only.
///
/// Buffers:
/// - row_ptrs: Input CSR row pointers (I32)
/// - col_indices: Input CSR column indices (I32)
/// - values: Input values
/// - l_row_ptrs: Output L row pointers (I32)
/// - l_col_indices: Output L column indices (I32)
/// - l_values: Output L values
/// - params: Uniform buffer with n (matrix size)
pub fn launch_split_lu_scatter_l(
    cache: &PipelineCache,
    queue: &Queue,
    row_ptrs: &Buffer,
    col_indices: &Buffer,
    values: &Buffer,
    l_row_ptrs: &Buffer,
    l_col_indices: &Buffer,
    l_values: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("split_lu_scatter_l_{}", suffix);

    let shader_source = generate_split_lu_scatter_l_shader(dtype)?;
    let module_name = format!("split_lu_scatter_l_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
    });
    let pipeline =
        cache.get_or_create_dynamic_pipeline("split_lu_scatter_l", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            row_ptrs,
            col_indices,
            values,
            l_row_ptrs,
            l_col_indices,
            l_values,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("split_lu_scatter_l"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("split_lu_scatter_l"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch split_lu scatter U kernel - scatters values into U matrix only.
///
/// Buffers:
/// - row_ptrs: Input CSR row pointers (I32)
/// - col_indices: Input CSR column indices (I32)
/// - values: Input values
/// - u_row_ptrs: Output U row pointers (I32)
/// - u_col_indices: Output U column indices (I32)
/// - u_values: Output U values
/// - params: Uniform buffer with n (matrix size)
pub fn launch_split_lu_scatter_u(
    cache: &PipelineCache,
    queue: &Queue,
    row_ptrs: &Buffer,
    col_indices: &Buffer,
    values: &Buffer,
    u_row_ptrs: &Buffer,
    u_col_indices: &Buffer,
    u_values: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("split_lu_scatter_u_{}", suffix);

    let shader_source = generate_split_lu_scatter_u_shader(dtype)?;
    let module_name = format!("split_lu_scatter_u_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
    });
    let pipeline =
        cache.get_or_create_dynamic_pipeline("split_lu_scatter_u", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            row_ptrs,
            col_indices,
            values,
            u_row_ptrs,
            u_col_indices,
            u_values,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("split_lu_scatter_u"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("split_lu_scatter_u"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Extract Lower Triangle Operations
// ============================================================================

/// Launch extract_lower count kernel - counts lower triangle non-zeros per row.
///
/// Buffers:
/// - row_ptrs: Input CSR row pointers (I32)
/// - col_indices: Input CSR column indices (I32)
/// - l_counts: Output L counts per row (I32)
/// - params: Uniform buffer with n (matrix size)
pub fn launch_extract_lower_count(
    cache: &PipelineCache,
    queue: &Queue,
    row_ptrs: &Buffer,
    col_indices: &Buffer,
    l_counts: &Buffer,
    params_buffer: &Buffer,
    n: usize,
) -> Result<()> {
    let shader_source = generate_extract_lower_count_shader();
    let module = cache.get_or_create_module_from_source("extract_lower_count", &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(
        "extract_lower_count",
        "extract_lower_count",
        &module,
        &layout,
    );

    let bind_group =
        cache.create_bind_group(&layout, &[row_ptrs, col_indices, l_counts, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("extract_lower_count"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("extract_lower_count"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch extract_lower scatter kernel - scatters lower triangle values.
///
/// Buffers:
/// - row_ptrs: Input CSR row pointers (I32)
/// - col_indices: Input CSR column indices (I32)
/// - values: Input values
/// - l_row_ptrs: Output L row pointers (I32)
/// - l_col_indices: Output L column indices (I32)
/// - l_values: Output L values
/// - params: Uniform buffer with n (matrix size)
pub fn launch_extract_lower_scatter(
    cache: &PipelineCache,
    queue: &Queue,
    row_ptrs: &Buffer,
    col_indices: &Buffer,
    values: &Buffer,
    l_row_ptrs: &Buffer,
    l_col_indices: &Buffer,
    l_values: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("extract_lower_scatter_{}", suffix);

    let shader_source = generate_extract_lower_scatter_shader(dtype)?;
    let module_name = format!("extract_lower_scatter_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_dynamic_pipeline(
        "extract_lower_scatter",
        &entry_point,
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            row_ptrs,
            col_indices,
            values,
            l_row_ptrs,
            l_col_indices,
            l_values,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("extract_lower_scatter"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("extract_lower_scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
