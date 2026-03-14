//! WGSL kernel launchers for sparse matrix element-wise merge operations.
//!
//! Provides launchers for CSR/CSC format merge operations:
//! - Count kernels: count output nonzeros per row/column
//! - Compute kernels: perform merge and operation
//! - Exclusive scan: compute prefix sum for offsets

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

// Static WGSL shader sources
const SPARSE_MERGE_COUNT: &str = include_str!("sparse_merge_count.wgsl");
const SPARSE_MERGE_F32: &str = include_str!("sparse_merge_f32.wgsl");
const SPARSE_MERGE_I32: &str = include_str!("sparse_merge_i32.wgsl");
const SPARSE_MERGE_U32: &str = include_str!("sparse_merge_u32.wgsl");

/// Return (module_key, shader_source) for a dtype-specific merge shader.
fn typed_merge_shader(dtype: DType) -> Result<(&'static str, &'static str)> {
    match dtype {
        DType::F32 => Ok(("sparse_merge_f32", SPARSE_MERGE_F32)),
        DType::I32 => Ok(("sparse_merge_i32", SPARSE_MERGE_I32)),
        DType::U32 => Ok(("sparse_merge_u32", SPARSE_MERGE_U32)),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_merge (WebGPU)",
        }),
    }
}

/// Return the dtype suffix string for entry point names.
fn dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::I32 => Ok("i32"),
        DType::U32 => Ok("u32"),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_merge (WebGPU)",
        }),
    }
}

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
    let module = cache.get_or_create_module("sparse_merge_count", SPARSE_MERGE_COUNT);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5, // a_row_ptrs, a_col_indices, b_row_ptrs, b_col_indices, row_counts
        num_uniform_buffers: 1, // params
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_pipeline("sparse_merge_count", "csr_merge_count", &module, &layout);

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
    let module = cache.get_or_create_module("sparse_merge_count", SPARSE_MERGE_COUNT);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_pipeline("sparse_merge_count", "csr_mul_count", &module, &layout);

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
    let (module_key, shader) = typed_merge_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point: &'static str = match suffix {
        "f32" => "csr_add_compute_f32",
        "i32" => "csr_add_compute_i32",
        "u32" => "csr_add_compute_u32",
        _ => unreachable!(),
    };

    let module = cache.get_or_create_module(module_key, shader);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9, // 3 for A, 3 for B, 3 for output
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

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
    let (module_key, shader) = typed_merge_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point: &'static str = match suffix {
        "f32" => "csr_sub_compute_f32",
        "i32" => "csr_sub_compute_i32",
        "u32" => "csr_sub_compute_u32",
        _ => unreachable!(),
    };

    let module = cache.get_or_create_module(module_key, shader);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

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
    let (module_key, shader) = typed_merge_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point: &'static str = match suffix {
        "f32" => "csr_mul_compute_f32",
        "i32" => "csr_mul_compute_i32",
        "u32" => "csr_mul_compute_u32",
        _ => unreachable!(),
    };

    let module = cache.get_or_create_module(module_key, shader);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

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
    let (module_key, shader) = typed_merge_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point: &'static str = match suffix {
        "f32" => "csr_div_compute_f32",
        "i32" => "csr_div_compute_i32",
        "u32" => "csr_div_compute_u32",
        _ => unreachable!(),
    };

    let module = cache.get_or_create_module(module_key, shader);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

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
    let module = cache.get_or_create_module("sparse_merge_count", SPARSE_MERGE_COUNT);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_pipeline("sparse_merge_count", "csc_merge_count", &module, &layout);

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
    let module = cache.get_or_create_module("sparse_merge_count", SPARSE_MERGE_COUNT);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        cache.get_or_create_pipeline("sparse_merge_count", "csc_mul_count", &module, &layout);

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
    let (module_key, shader) = typed_merge_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point: &'static str = match suffix {
        "f32" => "csc_add_compute_f32",
        "i32" => "csc_add_compute_i32",
        "u32" => "csc_add_compute_u32",
        _ => unreachable!(),
    };

    let module = cache.get_or_create_module(module_key, shader);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

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
    let (module_key, shader) = typed_merge_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point: &'static str = match suffix {
        "f32" => "csc_sub_compute_f32",
        "i32" => "csc_sub_compute_i32",
        "u32" => "csc_sub_compute_u32",
        _ => unreachable!(),
    };

    let module = cache.get_or_create_module(module_key, shader);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

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
    let (module_key, shader) = typed_merge_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point: &'static str = match suffix {
        "f32" => "csc_mul_compute_f32",
        "i32" => "csc_mul_compute_i32",
        "u32" => "csc_mul_compute_u32",
        _ => unreachable!(),
    };

    let module = cache.get_or_create_module(module_key, shader);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

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
    let (module_key, shader) = typed_merge_shader(dtype)?;
    let suffix = dtype_suffix(dtype)?;
    let entry_point: &'static str = match suffix {
        "f32" => "csc_div_compute_f32",
        "i32" => "csc_div_compute_i32",
        "u32" => "csc_div_compute_u32",
        _ => unreachable!(),
    };

    let module = cache.get_or_create_module(module_key, shader);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 9,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

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
    let module = cache.get_or_create_module("sparse_merge_count", SPARSE_MERGE_COUNT);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 1,
    });

    let pipeline =
        cache.get_or_create_pipeline("sparse_merge_count", "exclusive_scan_i32", &module, &layout);

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
