//! Kernel launchers for sparse format conversion operations.
//!
//! Provides functions to dispatch WGSL compute shaders for:
//! - CSR/CSC → COO (pointer expansion)
//! - COO → CSR/CSC (histogram + scan + scatter)
//! - CSR ↔ CSC (direct transpose)

use wgpu::{Buffer, Queue};

use super::generator::dtype_suffix;
use super::generator::sparse_conversions::{
    generate_coo_to_csc_scatter_shader, generate_coo_to_csr_scatter_shader,
    generate_copy_ptrs_shader, generate_csc_to_csr_scatter_shader,
    generate_csr_to_csc_scatter_shader, generate_expand_col_ptrs_shader,
    generate_expand_row_ptrs_shader, generate_histogram_shader,
};
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::Result;

/// Launch kernel to expand CSR row_ptrs to explicit row_indices.
pub fn launch_expand_row_ptrs(
    cache: &PipelineCache,
    queue: &Queue,
    row_ptrs: &Buffer,
    row_indices: &Buffer,
    params: &Buffer,
    nrows: usize,
) -> Result<()> {
    let source = generate_expand_row_ptrs_shader()?;
    let module = cache.get_or_create_module_from_source("expand_row_ptrs", &source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2, // row_ptrs, row_indices
        num_uniform_buffers: 1, // params
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_dynamic_pipeline(
        "expand_row_ptrs",
        "expand_row_ptrs",
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(&layout, &[row_ptrs, row_indices, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("expand_row_ptrs"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("expand_row_ptrs"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nrows), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch kernel to expand CSC col_ptrs to explicit col_indices.
pub fn launch_expand_col_ptrs(
    cache: &PipelineCache,
    queue: &Queue,
    col_ptrs: &Buffer,
    col_indices: &Buffer,
    params: &Buffer,
    ncols: usize,
) -> Result<()> {
    let source = generate_expand_col_ptrs_shader()?;
    let module = cache.get_or_create_module_from_source("expand_col_ptrs", &source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2, // col_ptrs, col_indices
        num_uniform_buffers: 1, // params
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_dynamic_pipeline(
        "expand_col_ptrs",
        "expand_col_ptrs",
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(&layout, &[col_ptrs, col_indices, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("expand_col_ptrs"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("expand_col_ptrs"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(ncols), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch histogram kernel to count elements per bucket.
pub fn launch_histogram(
    cache: &PipelineCache,
    queue: &Queue,
    indices: &Buffer,
    counts: &Buffer,
    params: &Buffer,
    nnz: usize,
) -> Result<()> {
    let source = generate_histogram_shader()?;
    let module = cache.get_or_create_module_from_source("histogram", &source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2, // indices, counts
        num_uniform_buffers: 1, // params
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_dynamic_pipeline("histogram", "histogram", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[indices, counts, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("histogram"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("histogram"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nnz), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch COO→CSR scatter kernel.
pub fn launch_coo_to_csr_scatter(
    cache: &PipelineCache,
    queue: &Queue,
    in_row_indices: &Buffer,
    in_col_indices: &Buffer,
    in_values: &Buffer,
    row_ptrs_atomic: &Buffer,
    out_col_indices: &Buffer,
    out_values: &Buffer,
    params: &Buffer,
    nnz: usize,
    dtype: DType,
) -> Result<()> {
    let source = generate_coo_to_csr_scatter_shader(dtype)?;
    let key = format!("coo_to_csr_scatter_{}", dtype_suffix(dtype)?);
    let module = cache.get_or_create_module_from_source(&key, &source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6, // in_row, in_col, in_val, row_ptrs_atomic, out_col, out_val
        num_uniform_buffers: 1, // params
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline(&key, "coo_to_csr_scatter", &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            in_row_indices,
            in_col_indices,
            in_values,
            row_ptrs_atomic,
            out_col_indices,
            out_values,
            params,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("coo_to_csr_scatter"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("coo_to_csr_scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nnz), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch COO→CSC scatter kernel.
pub fn launch_coo_to_csc_scatter(
    cache: &PipelineCache,
    queue: &Queue,
    in_row_indices: &Buffer,
    in_col_indices: &Buffer,
    in_values: &Buffer,
    col_ptrs_atomic: &Buffer,
    out_row_indices: &Buffer,
    out_values: &Buffer,
    params: &Buffer,
    nnz: usize,
    dtype: DType,
) -> Result<()> {
    let source = generate_coo_to_csc_scatter_shader(dtype)?;
    let key = format!("coo_to_csc_scatter_{}", dtype_suffix(dtype)?);
    let module = cache.get_or_create_module_from_source(&key, &source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline(&key, "coo_to_csc_scatter", &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            in_row_indices,
            in_col_indices,
            in_values,
            col_ptrs_atomic,
            out_row_indices,
            out_values,
            params,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("coo_to_csc_scatter"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("coo_to_csc_scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nnz), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch CSR→CSC transpose scatter kernel.
pub fn launch_csr_to_csc_scatter(
    cache: &PipelineCache,
    queue: &Queue,
    in_row_ptrs: &Buffer,
    in_col_indices: &Buffer,
    in_values: &Buffer,
    col_ptrs_atomic: &Buffer,
    out_row_indices: &Buffer,
    out_values: &Buffer,
    params: &Buffer,
    nrows: usize,
    dtype: DType,
) -> Result<()> {
    let source = generate_csr_to_csc_scatter_shader(dtype)?;
    let key = format!("csr_to_csc_scatter_{}", dtype_suffix(dtype)?);
    let module = cache.get_or_create_module_from_source(&key, &source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline(&key, "csr_to_csc_scatter", &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            in_row_ptrs,
            in_col_indices,
            in_values,
            col_ptrs_atomic,
            out_row_indices,
            out_values,
            params,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csr_to_csc_scatter"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csr_to_csc_scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nrows), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch CSC→CSR transpose scatter kernel.
pub fn launch_csc_to_csr_scatter(
    cache: &PipelineCache,
    queue: &Queue,
    in_col_ptrs: &Buffer,
    in_row_indices: &Buffer,
    in_values: &Buffer,
    row_ptrs_atomic: &Buffer,
    out_col_indices: &Buffer,
    out_values: &Buffer,
    params: &Buffer,
    ncols: usize,
    dtype: DType,
) -> Result<()> {
    let source = generate_csc_to_csr_scatter_shader(dtype)?;
    let key = format!("csc_to_csr_scatter_{}", dtype_suffix(dtype)?);
    let module = cache.get_or_create_module_from_source(&key, &source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline(&key, "csc_to_csr_scatter", &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            in_col_ptrs,
            in_row_indices,
            in_values,
            row_ptrs_atomic,
            out_col_indices,
            out_values,
            params,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csc_to_csr_scatter"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csc_to_csr_scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(ncols), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch kernel to copy pointers array.
pub fn launch_copy_ptrs(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    dst: &Buffer,
    params: &Buffer,
    n: usize,
) -> Result<()> {
    let source = generate_copy_ptrs_shader()?;
    let module = cache.get_or_create_module_from_source("copy_ptrs", &source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2, // src, dst
        num_uniform_buffers: 1, // params
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_dynamic_pipeline("copy_ptrs", "copy_ptrs", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, dst, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_ptrs"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("copy_ptrs"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch kernel for CSR to dense conversion.
pub fn launch_csr_to_dense(
    cache: &PipelineCache,
    queue: &Queue,
    row_ptrs: &Buffer,
    col_indices: &Buffer,
    values: &Buffer,
    dense: &Buffer,
    params: &Buffer,
    nrows: usize,
    dtype: DType,
) -> Result<()> {
    let source = super::generator::generate_csr_to_dense_shader(dtype)?;
    let key = format!("csr_to_dense_{}", dtype_suffix(dtype)?);
    let module = cache.get_or_create_module_from_source(&key, &source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4, // row_ptrs, col_indices, values, dense
        num_uniform_buffers: 1, // params
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_dynamic_pipeline(&key, "csr_to_dense", &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[row_ptrs, col_indices, values, dense, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("csr_to_dense"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("csr_to_dense"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nrows), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch kernel to count non-zeros in dense matrix.
pub fn launch_count_nonzeros(
    cache: &PipelineCache,
    queue: &Queue,
    dense: &Buffer,
    count: &Buffer,
    params: &Buffer,
    total_elems: usize,
    dtype: DType,
) -> Result<()> {
    let source = super::generator::generate_count_nonzeros_shader(dtype)?;
    let key = format!("count_nonzeros_{}", dtype_suffix(dtype)?);
    let module = cache.get_or_create_module_from_source(&key, &source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2, // dense, count
        num_uniform_buffers: 1, // params
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_dynamic_pipeline(&key, "count_nonzeros", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[dense, count, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("count_nonzeros"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("count_nonzeros"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elems), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch kernel for dense to COO scatter.
pub fn launch_dense_to_coo_scatter(
    cache: &PipelineCache,
    queue: &Queue,
    dense: &Buffer,
    row_indices: &Buffer,
    col_indices: &Buffer,
    values: &Buffer,
    write_pos: &Buffer,
    params: &Buffer,
    total_elems: usize,
    dtype: DType,
) -> Result<()> {
    let source = super::generator::generate_dense_to_coo_scatter_shader(dtype)?;
    let key = format!("dense_to_coo_scatter_{}", dtype_suffix(dtype)?);
    let module = cache.get_or_create_module_from_source(&key, &source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5, // dense, row_indices, col_indices, values, write_pos
        num_uniform_buffers: 1, // params
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_dynamic_pipeline(&key, "dense_to_coo_scatter", &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[dense, row_indices, col_indices, values, write_pos, params],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dense_to_coo_scatter"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dense_to_coo_scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elems), 1, 1);
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
    fn test_all_conversion_shaders_valid() {
        // Validate all generated shaders are syntactically correct
        validate_wgsl_syntax(&generate_expand_row_ptrs_shader().unwrap()).unwrap();
        validate_wgsl_syntax(&generate_expand_col_ptrs_shader().unwrap()).unwrap();
        validate_wgsl_syntax(&generate_histogram_shader().unwrap()).unwrap();
        validate_wgsl_syntax(&generate_copy_ptrs_shader().unwrap()).unwrap();

        for dtype in [DType::F32, DType::I32, DType::U32] {
            validate_wgsl_syntax(&generate_coo_to_csr_scatter_shader(dtype).unwrap()).unwrap();
            validate_wgsl_syntax(&generate_coo_to_csc_scatter_shader(dtype).unwrap()).unwrap();
            validate_wgsl_syntax(&generate_csr_to_csc_scatter_shader(dtype).unwrap()).unwrap();
            validate_wgsl_syntax(&generate_csc_to_csr_scatter_shader(dtype).unwrap()).unwrap();
        }
    }
}
