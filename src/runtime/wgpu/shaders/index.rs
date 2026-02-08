//! Index operation WGSL kernel launchers
//!
//! Provides launchers for indexing operations including:
//! - gather: Select elements along a dimension using an index tensor
//! - scatter: Scatter values to positions specified by an index tensor
//! - index_select: Select elements along a dimension using a 1D index tensor
//! - masked_select: Select elements where mask is true (returns flattened 1D tensor)
//! - masked_fill: Fill elements where mask is true with a scalar value
//!
//! All operations run entirely on GPU with no CPU fallback.

use wgpu::{Buffer, Queue};

use super::generator::{
    generate_embedding_lookup_shader, generate_gather_shader, generate_index_put_shader,
    generate_index_select_shader, generate_masked_fill_shader, generate_masked_select_shader,
    generate_scatter_shader, generate_validate_indices_shader,
};
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if dtype is supported for index operations on WebGPU.
fn check_dtype_supported(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::I32 | DType::U32 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Get the static module/entry point name for an index operation.
///
/// Returns the kernel name in format `{op}_{dtype_suffix}`.
/// For WebGPU index operations, module name and entry point are identical.
fn kernel_name(op: &'static str, dtype: DType) -> Result<&'static str> {
    match (op, dtype) {
        ("index_select", DType::F32) => Ok("index_select_f32"),
        ("index_select", DType::I32) => Ok("index_select_i32"),
        ("index_select", DType::U32) => Ok("index_select_u32"),
        ("index_put", DType::F32) => Ok("index_put_f32"),
        ("index_put", DType::I32) => Ok("index_put_i32"),
        ("index_put", DType::U32) => Ok("index_put_u32"),
        ("gather", DType::F32) => Ok("gather_f32"),
        ("gather", DType::I32) => Ok("gather_i32"),
        ("gather", DType::U32) => Ok("gather_u32"),
        ("scatter", DType::F32) => Ok("scatter_f32"),
        ("scatter", DType::I32) => Ok("scatter_i32"),
        ("scatter", DType::U32) => Ok("scatter_u32"),
        ("copy", DType::F32) => Ok("copy_f32"),
        ("copy", DType::I32) => Ok("copy_i32"),
        ("copy", DType::U32) => Ok("copy_u32"),
        ("masked_fill", DType::F32) => Ok("masked_fill_f32"),
        ("masked_fill", DType::I32) => Ok("masked_fill_i32"),
        ("masked_fill", DType::U32) => Ok("masked_fill_u32"),
        ("masked_select", DType::F32) => Ok("masked_select_f32"),
        ("masked_select", DType::I32) => Ok("masked_select_i32"),
        ("masked_select", DType::U32) => Ok("masked_select_u32"),
        ("embedding_lookup", DType::F32) => Ok("embedding_lookup_f32"),
        ("embedding_lookup", DType::I32) => Ok("embedding_lookup_i32"),
        ("embedding_lookup", DType::U32) => Ok("embedding_lookup_u32"),
        ("gather_nd", DType::F32) => Ok("gather_nd_f32"),
        ("gather_nd", DType::I32) => Ok("gather_nd_i32"),
        ("gather_nd", DType::U32) => Ok("gather_nd_u32"),
        ("bincount", DType::F32) => Ok("bincount_weighted_f32"),
        ("bincount", DType::I32) => Ok("bincount_weighted_i32"),
        ("bincount", DType::U32) => Ok("bincount_weighted_u32"),
        ("bincount_unweighted", _) => Ok("bincount_i32"),
        ("scatter_reduce_sum", DType::F32) => Ok("scatter_reduce_sum_f32"),
        ("scatter_reduce_sum", DType::I32) => Ok("scatter_reduce_sum_i32"),
        ("scatter_reduce_sum", DType::U32) => Ok("scatter_reduce_sum_u32"),
        ("scatter_reduce_max", DType::F32) => Ok("scatter_reduce_max_f32"),
        ("scatter_reduce_max", DType::I32) => Ok("scatter_reduce_max_i32"),
        ("scatter_reduce_max", DType::U32) => Ok("scatter_reduce_max_u32"),
        ("scatter_reduce_min", DType::F32) => Ok("scatter_reduce_min_f32"),
        ("scatter_reduce_min", DType::I32) => Ok("scatter_reduce_min_i32"),
        ("scatter_reduce_min", DType::U32) => Ok("scatter_reduce_min_u32"),
        ("gather_2d", DType::F32) => Ok("gather_2d_f32"),
        ("gather_2d", DType::I32) => Ok("gather_2d_i32"),
        ("gather_2d", DType::U32) => Ok("gather_2d_u32"),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

// ============================================================================
// Index Select Operation
// ============================================================================

/// Launch an index_select operation kernel.
///
/// Selects elements from input along the specified dimension using indices.
/// Output shape is the same as input except the dimension size becomes index_len.
pub fn launch_index_select(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    indices: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_output: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "index_select")?;

    let name = kernel_name("index_select", dtype)?;
    let shader_source = generate_index_select_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, indices, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("index_select"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("index_select"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_output), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Index Put Operation
// ============================================================================

/// Launch an index_put operation kernel.
///
/// Puts values from src at positions specified by indices along the dimension.
/// Output should be pre-initialized with a copy of the input tensor.
pub fn launch_index_put(
    cache: &PipelineCache,
    queue: &Queue,
    indices: &Buffer,
    src: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_src: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "index_put")?;

    let name = kernel_name("index_put", dtype)?;
    let shader_source = generate_index_put_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[indices, src, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("index_put"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("index_put"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_src), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Index Bounds Validation
// ============================================================================

/// Launch index bounds validation kernel.
///
/// Validates that all indices are within bounds [0, dim_size).
/// Returns the count of out-of-bounds indices in error_count buffer.
/// The error_count buffer must be initialized to 0 before calling.
pub fn launch_validate_indices(
    cache: &PipelineCache,
    queue: &Queue,
    indices: &Buffer,
    error_count: &Buffer,
    params_buffer: &Buffer,
    index_len: usize,
) -> Result<()> {
    if index_len == 0 {
        return Ok(());
    }

    let name = "validate_indices";
    let shader_source = generate_validate_indices_shader();
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[indices, error_count, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("validate_indices"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("validate_indices"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(index_len), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Gather Operation
// ============================================================================

/// Launch a gather operation kernel.
///
/// Gathers elements from input using indices along the specified dimension.
pub fn launch_gather(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    indices: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_elements: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "gather")?;

    let name = kernel_name("gather", dtype)?;
    let shader_source = generate_gather_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, indices, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gather"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gather"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Scatter Operation
// ============================================================================

/// Launch a copy operation kernel (for scatter initialization).
pub fn launch_copy(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "copy")?;

    // Copy kernel is defined in the scatter shader module
    let mod_name = kernel_name("scatter", dtype)?;
    let entry_point = kernel_name("copy", dtype)?;

    let shader_source = generate_scatter_shader(dtype)?;
    let module = cache.get_or_create_module(mod_name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(mod_name, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, dst, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("copy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch a scatter operation kernel.
///
/// Scatters values from src to output at positions specified by indices along dim.
pub fn launch_scatter(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    indices: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    src_total: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "scatter")?;

    let name = kernel_name("scatter", dtype)?;
    let shader_source = generate_scatter_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, indices, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scatter"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(src_total), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Masked Fill Operation
// ============================================================================

/// Launch a masked_fill operation kernel.
///
/// Fills elements in output with fill_value where mask is non-zero.
pub fn launch_masked_fill(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    mask: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "masked_fill")?;

    let name = kernel_name("masked_fill", dtype)?;
    let shader_source = generate_masked_fill_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, mask, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("masked_fill"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("masked_fill"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Masked Select Operation (3-phase)
// ============================================================================

/// Launch the masked_count phase of masked_select.
///
/// Counts the number of elements where mask is non-zero using atomic operations.
pub fn launch_masked_count(
    cache: &PipelineCache,
    queue: &Queue,
    mask: &Buffer,
    count_result: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "masked_count")?;

    let mod_name = kernel_name("masked_select", dtype)?;
    let shader_source = generate_masked_select_shader(dtype)?;
    let module = cache.get_or_create_module(mod_name, &shader_source);

    // For count: mask (read), count_result (atomic), params
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(mod_name, "masked_count", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[mask, count_result, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("masked_count"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("masked_count"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch the masked_prefix_sum phase of masked_select.
///
/// Computes exclusive prefix sum of mask for determining output positions.
pub fn launch_masked_prefix_sum(
    cache: &PipelineCache,
    queue: &Queue,
    mask: &Buffer,
    prefix_sum: &Buffer,
    params_buffer: &Buffer,
    _numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "masked_prefix_sum")?;

    let mod_name = kernel_name("masked_select", dtype)?;
    let shader_source = generate_masked_select_shader(dtype)?;
    let module = cache.get_or_create_module(mod_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(mod_name, "masked_prefix_sum", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[mask, prefix_sum, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("masked_prefix_sum"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("masked_prefix_sum"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // Sequential kernel - only 1 workgroup needed
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch the masked_select gather phase.
///
/// Gathers elements from input where mask is non-zero into output.
pub fn launch_masked_select(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    mask: &Buffer,
    prefix_sum: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "masked_select")?;

    let mod_name = kernel_name("masked_select", dtype)?;
    let entry_point = kernel_name("masked_select", dtype)?;

    let shader_source = generate_masked_select_shader(dtype)?;
    let module = cache.get_or_create_module(mod_name, &shader_source);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(mod_name, entry_point, &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[input, mask, prefix_sum, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("masked_select"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("masked_select"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Embedding Lookup Operation
// ============================================================================

/// Launch a gather_nd operation kernel.
///
/// Gathers slices from input using N-dimensional indices.
/// Input: input tensor, indices [num_slices, index_depth]
/// Output: output [num_slices, slice_size]
pub fn launch_gather_nd(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    indices: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_output: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "gather_nd")?;

    let name = kernel_name("gather_nd", dtype)?;
    let shader_source = super::generator::generate_gather_nd_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, indices, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gather_nd"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gather_nd"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_output), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Bincount Operation
// ============================================================================

/// Launch a bincount operation kernel.
///
/// Counts occurrences of each value in an integer tensor.
/// Input: integer tensor with values in [0, minlength)
/// Output: count tensor of shape [minlength]
pub fn launch_bincount(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    weights: Option<&Buffer>,
    output: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    weights_dtype: Option<DType>,
) -> Result<()> {
    let (name, shader_source) = if let Some(dtype) = weights_dtype {
        let name = kernel_name("bincount", dtype)?;
        let source = super::generator::generate_bincount_shader(Some(dtype))?;
        (name, source)
    } else {
        let name = kernel_name("bincount_unweighted", DType::I32)?;
        let source = super::generator::generate_bincount_shader(None)?;
        (name, source)
    };

    let module = cache.get_or_create_module(name, &shader_source);

    let (layout, bind_group) = if let Some(weights_buf) = weights {
        let layout = cache.get_or_create_layout(LayoutKey {
            num_storage_buffers: 3,
            num_uniform_buffers: 1,
            num_readonly_storage: 0,
        });
        let bind_group =
            cache.create_bind_group(&layout, &[input, weights_buf, output, params_buffer]);
        (layout, bind_group)
    } else {
        let layout = cache.get_or_create_layout(LayoutKey {
            num_storage_buffers: 2,
            num_uniform_buffers: 1,
            num_readonly_storage: 0,
        });
        let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);
        (layout, bind_group)
    };

    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bincount"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bincount"),
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
// Scatter Reduce Operation
// ============================================================================

/// Launch a scatter_reduce operation kernel.
///
/// Scatters values with reduction (sum, max, min).
/// Uses atomic operations for thread-safe accumulation.
pub fn launch_scatter_reduce(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    indices: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    total_src: usize,
    dtype: DType,
    op: &str,
) -> Result<()> {
    check_dtype_supported(dtype, "scatter_reduce")?;

    // Get static kernel name based on op type
    let op_name: &'static str = match op {
        "sum" => "scatter_reduce_sum",
        "max" => "scatter_reduce_max",
        "min" => "scatter_reduce_min",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "op",
                reason: format!("scatter_reduce op must be sum, max, or min, got {}", op),
            });
        }
    };

    let name = kernel_name(op_name, dtype)?;
    let shader_source = super::generator::generate_scatter_reduce_shader(dtype, op)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, indices, dst, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scatter_reduce"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter_reduce"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_src), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Embedding Lookup Operation
// ============================================================================

/// Launch an embedding_lookup operation kernel.
///
/// Looks up embeddings from a 2D embedding table using indices.
/// Input: embeddings [vocab_size, embedding_dim], indices [num_indices]
/// Output: output [num_indices, embedding_dim]
///
/// This is the industry-standard embedding lookup operation used in neural networks
/// for word embeddings, entity embeddings, etc.
pub fn launch_embedding_lookup(
    cache: &PipelineCache,
    queue: &Queue,
    embeddings: &Buffer,
    indices: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    num_indices: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "embedding_lookup")?;

    let name = kernel_name("embedding_lookup", dtype)?;
    let shader_source = generate_embedding_lookup_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[embeddings, indices, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("embedding_lookup"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("embedding_lookup"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(num_indices), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Gather 2D Operation
// ============================================================================

/// Launch a gather_2d operation kernel.
///
/// Gathers elements from a 2D matrix at specific (row, col) positions.
/// Input: input [nrows, ncols], rows [num_indices], cols [num_indices]
/// Output: output [num_indices]
///
/// For each index i: output[i] = input[rows[i], cols[i]]
#[allow(clippy::too_many_arguments)]
pub fn launch_gather_2d(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    rows: &Buffer,
    cols: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    num_indices: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "gather_2d")?;

    let name = kernel_name("gather_2d", dtype)?;
    let shader_source = super::generator::generate_gather_2d_shader(dtype)?;
    let module = cache.get_or_create_module(name, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, rows, cols, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gather_2d"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gather_2d"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(num_indices), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
