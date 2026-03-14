//! Sparse linear algebra WGSL kernel launchers
//!
//! Provides launchers for sparse matrix filtering operations:
//! - `split_lu_count` - Count L and U non-zeros per row
//! - `split_lu_scatter` - Scatter values into L and U matrices
//! - `extract_lower_count` - Count lower triangle non-zeros per row
//! - `extract_lower_scatter` - Scatter lower triangle values

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

// Static WGSL shader sources
const SPARSE_LINALG: &str = include_str!("sparse_linalg.wgsl");
const SPARSE_LINALG_SPLIT_F32: &str = include_str!("sparse_linalg_split_f32.wgsl");

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
    let module = cache.get_or_create_module("sparse_linalg_split_f32", SPARSE_LINALG_SPLIT_F32);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "sparse_linalg_split_f32",
        "split_lu_count",
        &module,
        &layout,
    );

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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "split_lu_scatter_l (WebGPU)",
        });
    }

    let module = cache.get_or_create_module("sparse_linalg_split_f32", SPARSE_LINALG_SPLIT_F32);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "sparse_linalg_split_f32",
        "split_lu_scatter_l_f32",
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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "split_lu_scatter_u (WebGPU)",
        });
    }

    let module = cache.get_or_create_module("sparse_linalg_split_f32", SPARSE_LINALG_SPLIT_F32);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "sparse_linalg_split_f32",
        "split_lu_scatter_u_f32",
        &module,
        &layout,
    );

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
    let module = cache.get_or_create_module("sparse_linalg_split_f32", SPARSE_LINALG_SPLIT_F32);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "sparse_linalg_split_f32",
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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "extract_lower_scatter (WebGPU)",
        });
    }

    let module = cache.get_or_create_module("sparse_linalg_split_f32", SPARSE_LINALG_SPLIT_F32);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "sparse_linalg_split_f32",
        "extract_lower_scatter_f32",
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

// ============================================================================
// Sparse LU Primitive Operations
// ============================================================================

/// Sparse LU uniform params for operations that need scalar parameters
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SparseLuParams {
    /// Scaling factor for sparse operations
    pub scale: f32,
    /// Number of non-zero entries
    pub nnz: u32,
}

/// Launch sparse scatter kernel - f32
///
/// Scatters values into work vector: `work[row_indices[i]] = values[i]`
pub fn launch_sparse_scatter_f32(
    cache: &PipelineCache,
    queue: &Queue,
    values: &Buffer,
    row_indices: &Buffer,
    work: &Buffer,
    nnz: usize,
) -> Result<()> {
    let module = cache.get_or_create_module("sparse_linalg", SPARSE_LINALG);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 0,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("sparse_linalg", "sparse_scatter_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[values, row_indices, work]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sparse_scatter_f32"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sparse_scatter_f32"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nnz), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch sparse AXPY kernel - f32
///
/// Computes: `work[row_indices[i]] -= scale * values[i]`
pub fn launch_sparse_axpy_f32(
    cache: &PipelineCache,
    queue: &Queue,
    params_buffer: &Buffer,
    values: &Buffer,
    row_indices: &Buffer,
    work: &Buffer,
    nnz: usize,
) -> Result<()> {
    let module = cache.get_or_create_module("sparse_linalg", SPARSE_LINALG);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("sparse_linalg", "sparse_axpy_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[params_buffer, values, row_indices, work]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sparse_axpy_f32"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sparse_axpy_f32"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nnz), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch sparse gather and clear kernel - f32
///
/// Gathers: `output[i] = work[row_indices[i]]`, then clears `work[row_indices[i]] = 0`
pub fn launch_sparse_gather_clear_f32(
    cache: &PipelineCache,
    queue: &Queue,
    work: &Buffer,
    row_indices: &Buffer,
    output: &Buffer,
    nnz: usize,
) -> Result<()> {
    let module = cache.get_or_create_module("sparse_linalg", SPARSE_LINALG);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 0,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("sparse_linalg", "sparse_gather_clear_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[work, row_indices, output]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sparse_gather_clear_f32"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sparse_gather_clear_f32"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nnz), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Divide by pivot uniform params
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DividePivotParams {
    /// Inverse of the pivot value (1.0 / pivot)
    pub inv_pivot: f32,
    /// Number of non-zero entries to process
    pub nnz: u32,
}

/// Launch sparse divide by pivot kernel - f32
///
/// Computes: `work[row_indices[i]] *= inv_pivot`
pub fn launch_sparse_divide_pivot_f32(
    cache: &PipelineCache,
    queue: &Queue,
    params_buffer: &Buffer,
    work: &Buffer,
    row_indices: &Buffer,
    nnz: usize,
) -> Result<()> {
    let module = cache.get_or_create_module("sparse_linalg", SPARSE_LINALG);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("sparse_linalg", "sparse_divide_pivot_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[params_buffer, work, row_indices]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sparse_divide_pivot_f32"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sparse_divide_pivot_f32"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nnz), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch sparse clear kernel - f32
///
/// Clears: `work[row_indices[i]] = 0`
pub fn launch_sparse_clear_f32(
    cache: &PipelineCache,
    queue: &Queue,
    work: &Buffer,
    row_indices: &Buffer,
    nnz: usize,
) -> Result<()> {
    let module = cache.get_or_create_module("sparse_linalg", SPARSE_LINALG);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 0,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("sparse_linalg", "sparse_clear_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[work, row_indices]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sparse_clear_f32"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sparse_clear_f32"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nnz), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
