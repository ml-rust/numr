//! WGSL kernel launchers for sparse matrix element-wise merge operations.
//!
//! Provides launchers for CSR/CSC format merge operations:
//! - Count kernels: count output nonzeros per row/column
//! - Compute kernels: perform merge and operation
//! - Exclusive scan: compute prefix sum for offsets

use wgpu::{Buffer, Queue};

use super::generator::dtype_suffix;
use super::generator::sparse_merge::{
    generate_csc_add_compute_shader, generate_csc_div_compute_shader,
    generate_csc_merge_count_shader, generate_csc_mul_compute_shader,
    generate_csc_mul_count_shader, generate_csc_sub_compute_shader,
    generate_csr_add_compute_shader, generate_csr_div_compute_shader,
    generate_csr_merge_count_shader, generate_csr_mul_compute_shader,
    generate_csr_mul_count_shader, generate_csr_sub_compute_shader, generate_exclusive_scan_shader,
};
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::Result;

// ============================================================================
// CSR Count Kernels
// ============================================================================

/// Launch CSR merge count kernel (union semantics for add/sub)
pub fn launch_csr_merge_count(
    cache: &PipelineCache,
    queue: &Queue,
    a_row_ptrs: &Buffer,
    a_col_indices: &Buffer,
    b_row_ptrs: &Buffer,
    b_col_indices: &Buffer,
    row_counts: &Buffer,
    params_buffer: &Buffer,
    nrows: usize,
) -> Result<()> {
    let shader_source = generate_csr_merge_count_shader();
    let module = cache.get_or_create_module_from_source("csr_merge_count", &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5, // a_row_ptrs, a_col_indices, b_row_ptrs, b_col_indices, row_counts
        num_uniform_buffers: 1, // params
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_dynamic_pipeline(
        "csr_merge_count",
        "csr_merge_count",
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            a_row_ptrs,
            a_col_indices,
            b_row_ptrs,
            b_col_indices,
            row_counts,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csr_merge_count"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csr_merge_count"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nrows), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch CSR mul count kernel (intersection semantics)
pub fn launch_csr_mul_count(
    cache: &PipelineCache,
    queue: &Queue,
    a_row_ptrs: &Buffer,
    a_col_indices: &Buffer,
    b_row_ptrs: &Buffer,
    b_col_indices: &Buffer,
    row_counts: &Buffer,
    params_buffer: &Buffer,
    nrows: usize,
) -> Result<()> {
    let shader_source = generate_csr_mul_count_shader();
    let module = cache.get_or_create_module_from_source("csr_mul_count", &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline("csr_mul_count", "csr_mul_count", &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            a_row_ptrs,
            a_col_indices,
            b_row_ptrs,
            b_col_indices,
            row_counts,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csr_mul_count"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csr_mul_count"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nrows), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// CSR Compute Kernels
// ============================================================================

/// Launch CSR add compute kernel
pub fn launch_csr_add_compute(
    cache: &PipelineCache,
    queue: &Queue,
    a_row_ptrs: &Buffer,
    a_col_indices: &Buffer,
    a_values: &Buffer,
    b_row_ptrs: &Buffer,
    b_col_indices: &Buffer,
    b_values: &Buffer,
    out_row_ptrs: &Buffer,
    out_col_indices: &Buffer,
    out_values: &Buffer,
    params_buffer: &Buffer,
    nrows: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("csr_add_compute_{}", suffix);

    let shader_source = generate_csr_add_compute_shader(dtype)?;
    let module_name = format!("csr_add_compute_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9, // 3 for A, 3 for B, 3 for output
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline("csr_add_compute", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            out_row_ptrs,
            out_col_indices,
            out_values,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csr_add_compute"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csr_add_compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nrows), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch CSR sub compute kernel
pub fn launch_csr_sub_compute(
    cache: &PipelineCache,
    queue: &Queue,
    a_row_ptrs: &Buffer,
    a_col_indices: &Buffer,
    a_values: &Buffer,
    b_row_ptrs: &Buffer,
    b_col_indices: &Buffer,
    b_values: &Buffer,
    out_row_ptrs: &Buffer,
    out_col_indices: &Buffer,
    out_values: &Buffer,
    params_buffer: &Buffer,
    nrows: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("csr_sub_compute_{}", suffix);

    let shader_source = generate_csr_sub_compute_shader(dtype)?;
    let module_name = format!("csr_sub_compute_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline("csr_sub_compute", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            out_row_ptrs,
            out_col_indices,
            out_values,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csr_sub_compute"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csr_sub_compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nrows), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch CSR mul compute kernel
pub fn launch_csr_mul_compute(
    cache: &PipelineCache,
    queue: &Queue,
    a_row_ptrs: &Buffer,
    a_col_indices: &Buffer,
    a_values: &Buffer,
    b_row_ptrs: &Buffer,
    b_col_indices: &Buffer,
    b_values: &Buffer,
    out_row_ptrs: &Buffer,
    out_col_indices: &Buffer,
    out_values: &Buffer,
    params_buffer: &Buffer,
    nrows: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("csr_mul_compute_{}", suffix);

    let shader_source = generate_csr_mul_compute_shader(dtype)?;
    let module_name = format!("csr_mul_compute_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline("csr_mul_compute", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            out_row_ptrs,
            out_col_indices,
            out_values,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csr_mul_compute"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csr_mul_compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nrows), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch CSR div compute kernel
pub fn launch_csr_div_compute(
    cache: &PipelineCache,
    queue: &Queue,
    a_row_ptrs: &Buffer,
    a_col_indices: &Buffer,
    a_values: &Buffer,
    b_row_ptrs: &Buffer,
    b_col_indices: &Buffer,
    b_values: &Buffer,
    out_row_ptrs: &Buffer,
    out_col_indices: &Buffer,
    out_values: &Buffer,
    params_buffer: &Buffer,
    nrows: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("csr_div_compute_{}", suffix);

    let shader_source = generate_csr_div_compute_shader(dtype)?;
    let module_name = format!("csr_div_compute_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline("csr_div_compute", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            out_row_ptrs,
            out_col_indices,
            out_values,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csr_div_compute"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csr_div_compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nrows), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// CSC Count Kernels
// ============================================================================

/// Launch CSC merge count kernel (union semantics)
pub fn launch_csc_merge_count(
    cache: &PipelineCache,
    queue: &Queue,
    a_col_ptrs: &Buffer,
    a_row_indices: &Buffer,
    b_col_ptrs: &Buffer,
    b_row_indices: &Buffer,
    col_counts: &Buffer,
    params_buffer: &Buffer,
    ncols: usize,
) -> Result<()> {
    let shader_source = generate_csc_merge_count_shader();
    let module = cache.get_or_create_module_from_source("csc_merge_count", &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_dynamic_pipeline(
        "csc_merge_count",
        "csc_merge_count",
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            a_col_ptrs,
            a_row_indices,
            b_col_ptrs,
            b_row_indices,
            col_counts,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csc_merge_count"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csc_merge_count"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(ncols), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch CSC mul count kernel (intersection semantics)
pub fn launch_csc_mul_count(
    cache: &PipelineCache,
    queue: &Queue,
    a_col_ptrs: &Buffer,
    a_row_indices: &Buffer,
    b_col_ptrs: &Buffer,
    b_row_indices: &Buffer,
    col_counts: &Buffer,
    params_buffer: &Buffer,
    ncols: usize,
) -> Result<()> {
    let shader_source = generate_csc_mul_count_shader();
    let module = cache.get_or_create_module_from_source("csc_mul_count", &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline("csc_mul_count", "csc_mul_count", &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            a_col_ptrs,
            a_row_indices,
            b_col_ptrs,
            b_row_indices,
            col_counts,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csc_mul_count"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csc_mul_count"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(ncols), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// CSC Compute Kernels
// ============================================================================

/// Launch CSC add compute kernel
pub fn launch_csc_add_compute(
    cache: &PipelineCache,
    queue: &Queue,
    a_col_ptrs: &Buffer,
    a_row_indices: &Buffer,
    a_values: &Buffer,
    b_col_ptrs: &Buffer,
    b_row_indices: &Buffer,
    b_values: &Buffer,
    out_col_ptrs: &Buffer,
    out_row_indices: &Buffer,
    out_values: &Buffer,
    params_buffer: &Buffer,
    ncols: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("csc_add_compute_{}", suffix);

    let shader_source = generate_csc_add_compute_shader(dtype)?;
    let module_name = format!("csc_add_compute_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline("csc_add_compute", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            out_col_ptrs,
            out_row_indices,
            out_values,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csc_add_compute"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csc_add_compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(ncols), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch CSC sub compute kernel
pub fn launch_csc_sub_compute(
    cache: &PipelineCache,
    queue: &Queue,
    a_col_ptrs: &Buffer,
    a_row_indices: &Buffer,
    a_values: &Buffer,
    b_col_ptrs: &Buffer,
    b_row_indices: &Buffer,
    b_values: &Buffer,
    out_col_ptrs: &Buffer,
    out_row_indices: &Buffer,
    out_values: &Buffer,
    params_buffer: &Buffer,
    ncols: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("csc_sub_compute_{}", suffix);

    let shader_source = generate_csc_sub_compute_shader(dtype)?;
    let module_name = format!("csc_sub_compute_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline("csc_sub_compute", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            out_col_ptrs,
            out_row_indices,
            out_values,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csc_sub_compute"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csc_sub_compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(ncols), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch CSC mul compute kernel
pub fn launch_csc_mul_compute(
    cache: &PipelineCache,
    queue: &Queue,
    a_col_ptrs: &Buffer,
    a_row_indices: &Buffer,
    a_values: &Buffer,
    b_col_ptrs: &Buffer,
    b_row_indices: &Buffer,
    b_values: &Buffer,
    out_col_ptrs: &Buffer,
    out_row_indices: &Buffer,
    out_values: &Buffer,
    params_buffer: &Buffer,
    ncols: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("csc_mul_compute_{}", suffix);

    let shader_source = generate_csc_mul_compute_shader(dtype)?;
    let module_name = format!("csc_mul_compute_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline("csc_mul_compute", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            out_col_ptrs,
            out_row_indices,
            out_values,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csc_mul_compute"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csc_mul_compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(ncols), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch CSC div compute kernel
pub fn launch_csc_div_compute(
    cache: &PipelineCache,
    queue: &Queue,
    a_col_ptrs: &Buffer,
    a_row_indices: &Buffer,
    a_values: &Buffer,
    b_col_ptrs: &Buffer,
    b_row_indices: &Buffer,
    b_values: &Buffer,
    out_col_ptrs: &Buffer,
    out_row_indices: &Buffer,
    out_values: &Buffer,
    params_buffer: &Buffer,
    ncols: usize,
    dtype: DType,
) -> Result<()> {
    let suffix = dtype_suffix(dtype)?;
    let entry_point = format!("csc_div_compute_{}", suffix);

    let shader_source = generate_csc_div_compute_shader(dtype)?;
    let module_name = format!("csc_div_compute_{}", suffix);
    let module = cache.get_or_create_module_from_source(&module_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline("csc_div_compute", &entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            out_col_ptrs,
            out_row_indices,
            out_values,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csc_div_compute"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csc_div_compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(ncols), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Exclusive Scan
// ============================================================================

/// Launch exclusive scan kernel for i32 arrays
///
/// Computes prefix sum: `output[i] = sum(input[0..i])`
/// output has n+1 elements (`output[n]` = total sum)
pub fn launch_exclusive_scan_i32(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
) -> Result<()> {
    let shader_source = generate_exclusive_scan_shader();
    let module = cache.get_or_create_module_from_source("exclusive_scan_i32", &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 1,
    });

    let pipeline = cache.get_or_create_dynamic_pipeline(
        "exclusive_scan_i32",
        "exclusive_scan_i32",
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("exclusive_scan_i32"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("exclusive_scan_i32"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // Sequential scan - just one workgroup
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate_wgsl_syntax(source: &str) -> std::result::Result<(), String> {
        use wgpu::naga::front::wgsl;
        let mut frontend = wgsl::Frontend::new();
        frontend
            .parse(source)
            .map(|_| ())
            .map_err(|e| format!("WGSL parse error: {e}"))
    }

    #[test]
    fn test_generated_shaders_are_valid() {
        // Test all generated shaders have valid syntax
        validate_wgsl_syntax(&generate_csr_merge_count_shader())
            .expect("CSR merge count should be valid");
        validate_wgsl_syntax(&generate_csr_mul_count_shader())
            .expect("CSR mul count should be valid");
        validate_wgsl_syntax(&generate_csc_merge_count_shader())
            .expect("CSC merge count should be valid");
        validate_wgsl_syntax(&generate_csc_mul_count_shader())
            .expect("CSC mul count should be valid");
        validate_wgsl_syntax(&generate_exclusive_scan_shader())
            .expect("Exclusive scan should be valid");

        // Test compute shaders for F32
        validate_wgsl_syntax(&generate_csr_add_compute_shader(DType::F32).unwrap())
            .expect("CSR add compute should be valid");
        validate_wgsl_syntax(&generate_csr_sub_compute_shader(DType::F32).unwrap())
            .expect("CSR sub compute should be valid");
        validate_wgsl_syntax(&generate_csr_mul_compute_shader(DType::F32).unwrap())
            .expect("CSR mul compute should be valid");
        validate_wgsl_syntax(&generate_csr_div_compute_shader(DType::F32).unwrap())
            .expect("CSR div compute should be valid");
        validate_wgsl_syntax(&generate_csc_add_compute_shader(DType::F32).unwrap())
            .expect("CSC add compute should be valid");
    }
}
