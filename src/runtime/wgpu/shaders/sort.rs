//! Sort operation WGSL kernel launchers
//!
//! Provides launchers for sorting operations including:
//! - Sort, argsort (bitonic sort)
//! - Topk (top-k values and indices)
//! - Searchsorted (binary search)
//! - Nonzero (two-phase: count + gather)
//! - Unique (two-phase: count + extract on sorted input)
//!
//! Multi-dtype support: F32, I32, U32

use std::collections::HashMap;
use std::sync::RwLock;

use wgpu::{Buffer, Queue};

use super::generator::{
    generate_count_nonzero_shader, generate_flat_to_multi_index_shader,
    generate_gather_nonzero_shader, generate_searchsorted_shader, generate_sort_shader,
    generate_topk_shader, generate_unique_shader,
};
use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Shader Module Cache
// ============================================================================

static SORT_SHADER_CACHE: RwLock<Option<HashMap<(DType, &'static str), String>>> =
    RwLock::new(None);

fn get_shader(dtype: DType, op: &'static str) -> Result<String> {
    // Check cache
    {
        let cache = SORT_SHADER_CACHE.read().unwrap();
        if let Some(ref map) = *cache {
            if let Some(shader) = map.get(&(dtype, op)) {
                return Ok(shader.clone());
            }
        }
    }

    // Generate shader
    let shader = match op {
        "sort" => generate_sort_shader(dtype)?,
        "topk" => generate_topk_shader(dtype)?,
        "searchsorted" => generate_searchsorted_shader(dtype)?,
        "count_nonzero" => generate_count_nonzero_shader(dtype)?,
        "gather_nonzero" => generate_gather_nonzero_shader(dtype)?,
        "unique" => generate_unique_shader(dtype)?,
        "flat_to_multi_index" => generate_flat_to_multi_index_shader()?,
        _ => {
            return Err(Error::InvalidArgument {
                arg: "op",
                reason: format!("Unknown sort operation: {}", op),
            });
        }
    };

    // Cache and return
    {
        let mut cache = SORT_SHADER_CACHE.write().unwrap();
        let map = cache.get_or_insert_with(HashMap::new);
        map.insert((dtype, op), shader.clone());
    }
    Ok(shader)
}

fn module_key(dtype: DType, op: &'static str) -> String {
    let suffix = match dtype {
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::U32 => "u32",
        _ => "f32",
    };
    format!("{}_{}", op, suffix)
}

fn entry_point(op: &str, dtype: DType) -> String {
    let suffix = match dtype {
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::U32 => "u32",
        _ => "f32",
    };
    format!("{}_{}", op, suffix)
}

fn check_dtype_supported(dtype: DType, op: &'static str) -> Result<()> {
    if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

// ============================================================================
// Sort Operations
// ============================================================================

/// Launch sort with indices kernel
pub fn launch_sort(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    values_output: &Buffer,
    indices_output: &Buffer,
    params_buffer: &Buffer,
    outer_size: usize,
    inner_size: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "sort")?;

    let shader = get_shader(dtype, "sort")?;
    let module_name = module_key(dtype, "sort");
    let ep = entry_point("sort", dtype);

    let static_module: &'static str = Box::leak(module_name.into_boxed_str());
    let static_shader: &'static str = Box::leak(shader.into_boxed_str());
    let static_ep: &'static str = Box::leak(ep.into_boxed_str());

    let module = cache.get_or_create_module(static_module, static_shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(static_module, static_ep, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[input, values_output, indices_output, params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sort"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sort"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(outer_size as u32, inner_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch sort values only kernel (no indices)
pub fn launch_sort_values_only(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    outer_size: usize,
    inner_size: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "sort")?;

    let shader = get_shader(dtype, "sort")?;
    let module_name = module_key(dtype, "sort");
    let ep = entry_point("sort_values_only", dtype);

    let static_module: &'static str = Box::leak(module_name.into_boxed_str());
    let static_shader: &'static str = Box::leak(shader.into_boxed_str());
    let static_ep: &'static str = Box::leak(ep.into_boxed_str());

    let module = cache.get_or_create_module(static_module, static_shader);
    // Need a 4-buffer layout but only use 3 (input, output, dummy_indices, params)
    // Actually for values_only we need different layout
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(static_module, static_ep, &module, &layout);

    // Create dummy indices buffer for the binding
    let dummy_buf = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_indices"),
        size: 4, // minimum
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = cache.create_bind_group(&layout, &[input, output, &dummy_buf, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sort_values_only"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sort_values_only"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(outer_size as u32, inner_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch argsort kernel (indices only)
pub fn launch_argsort(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    indices_output: &Buffer,
    params_buffer: &Buffer,
    outer_size: usize,
    inner_size: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "argsort")?;

    let shader = get_shader(dtype, "sort")?;
    let module_name = module_key(dtype, "sort");
    let ep = entry_point("argsort", dtype);

    let static_module: &'static str = Box::leak(module_name.into_boxed_str());
    let static_shader: &'static str = Box::leak(shader.into_boxed_str());
    let static_ep: &'static str = Box::leak(ep.into_boxed_str());

    let module = cache.get_or_create_module(static_module, static_shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(static_module, static_ep, &module, &layout);

    // Create dummy values buffer for the binding
    let dummy_buf = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_values"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group =
        cache.create_bind_group(&layout, &[input, &dummy_buf, indices_output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("argsort"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("argsort"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(outer_size as u32, inner_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Topk Operations
// ============================================================================

/// Launch topk kernel
pub fn launch_topk(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    values_output: &Buffer,
    indices_output: &Buffer,
    params_buffer: &Buffer,
    outer_size: usize,
    inner_size: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "topk")?;

    let shader = get_shader(dtype, "topk")?;
    let module_name = module_key(dtype, "topk");
    let ep = entry_point("topk", dtype);

    let static_module: &'static str = Box::leak(module_name.into_boxed_str());
    let static_shader: &'static str = Box::leak(shader.into_boxed_str());
    let static_ep: &'static str = Box::leak(ep.into_boxed_str());

    let module = cache.get_or_create_module(static_module, static_shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(static_module, static_ep, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[input, values_output, indices_output, params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("topk"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("topk"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(outer_size as u32, inner_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Searchsorted Operations
// ============================================================================

/// Launch searchsorted kernel
pub fn launch_searchsorted(
    cache: &PipelineCache,
    queue: &Queue,
    sorted_seq: &Buffer,
    values: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    num_values: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "searchsorted")?;

    let shader = get_shader(dtype, "searchsorted")?;
    let module_name = module_key(dtype, "searchsorted");
    let ep = entry_point("searchsorted", dtype);

    let static_module: &'static str = Box::leak(module_name.into_boxed_str());
    let static_shader: &'static str = Box::leak(shader.into_boxed_str());
    let static_ep: &'static str = Box::leak(ep.into_boxed_str());

    let module = cache.get_or_create_module(static_module, static_shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(static_module, static_ep, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[sorted_seq, values, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("searchsorted"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("searchsorted"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(num_values), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Nonzero Operations (Two-phase)
// ============================================================================

/// Launch count_nonzero kernel (phase 1)
pub fn launch_count_nonzero(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    count_output: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "count_nonzero")?;

    let shader = get_shader(dtype, "count_nonzero")?;
    let module_name = module_key(dtype, "count_nonzero");
    let ep = entry_point("count_nonzero", dtype);

    let static_module: &'static str = Box::leak(module_name.into_boxed_str());
    let static_shader: &'static str = Box::leak(shader.into_boxed_str());
    let static_ep: &'static str = Box::leak(ep.into_boxed_str());

    let module = cache.get_or_create_module(static_module, static_shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(static_module, static_ep, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, count_output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("count_nonzero"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("count_nonzero"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch gather_nonzero kernel (phase 2)
pub fn launch_gather_nonzero(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    indices_output: &Buffer,
    counter: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "gather_nonzero")?;

    let shader = get_shader(dtype, "gather_nonzero")?;
    let module_name = module_key(dtype, "gather_nonzero");
    let ep = entry_point("gather_nonzero", dtype);

    let static_module: &'static str = Box::leak(module_name.into_boxed_str());
    let static_shader: &'static str = Box::leak(shader.into_boxed_str());
    let static_ep: &'static str = Box::leak(ep.into_boxed_str());

    let module = cache.get_or_create_module(static_module, static_shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(static_module, static_ep, &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[input, indices_output, counter, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gather_nonzero"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gather_nonzero"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch flat_to_multi_index kernel
pub fn launch_flat_to_multi_index(
    cache: &PipelineCache,
    queue: &Queue,
    flat_indices: &Buffer,
    multi_indices: &Buffer,
    params_buffer: &Buffer,
    nnz: usize,
) -> Result<()> {
    let shader = get_shader(DType::I32, "flat_to_multi_index")?;

    let static_module: &'static str = "flat_to_multi_index";
    let static_shader: &'static str = Box::leak(shader.into_boxed_str());

    let module = cache.get_or_create_module(static_module, static_shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
    });
    let pipeline =
        cache.get_or_create_pipeline(static_module, "flat_to_multi_index", &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[flat_indices, multi_indices, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat_to_multi_index"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flat_to_multi_index"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nnz), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Unique Operations (Two-phase)
// ============================================================================

/// Launch count_unique kernel (phase 1 - on sorted input)
pub fn launch_count_unique(
    cache: &PipelineCache,
    queue: &Queue,
    sorted_input: &Buffer,
    count_output: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "unique")?;

    let shader = get_shader(dtype, "unique")?;
    let module_name = module_key(dtype, "unique");
    let ep = entry_point("count_unique", dtype);

    let static_module: &'static str = Box::leak(module_name.into_boxed_str());
    let static_shader: &'static str = Box::leak(shader.into_boxed_str());
    let static_ep: &'static str = Box::leak(ep.into_boxed_str());

    let module = cache.get_or_create_module(static_module, static_shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(static_module, static_ep, &module, &layout);

    // Create dummy output buffer for the binding
    let dummy_buf = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_unique_output"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = cache.create_bind_group(
        &layout,
        &[sorted_input, &dummy_buf, count_output, params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("count_unique"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("count_unique"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch extract_unique kernel (phase 2 - on sorted input)
pub fn launch_extract_unique(
    cache: &PipelineCache,
    queue: &Queue,
    sorted_input: &Buffer,
    unique_output: &Buffer,
    counter: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_supported(dtype, "unique")?;

    let shader = get_shader(dtype, "unique")?;
    let module_name = module_key(dtype, "unique");
    let ep = entry_point("extract_unique", dtype);

    let static_module: &'static str = Box::leak(module_name.into_boxed_str());
    let static_shader: &'static str = Box::leak(shader.into_boxed_str());
    let static_ep: &'static str = Box::leak(ep.into_boxed_str());

    let module = cache.get_or_create_module(static_module, static_shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(static_module, static_ep, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[sorted_input, unique_output, counter, params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("extract_unique"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("extract_unique"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
