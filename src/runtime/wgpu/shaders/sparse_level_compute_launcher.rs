//! Sparse level computation kernel launchers for WebGPU
//!
//! Provides GPU-native level computation for level-synchronous sparse factorization.
//! Avoids GPU↔CPU transfers for CSR structure analysis.

use std::sync::Arc;

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::Result;

// ============================================================================
// i64 → i32 casting
// ============================================================================

/// Cast i64 CSR indices to i32 on GPU.
///
/// WebGPU doesn't support i64 compute types. CSR indices are stored as i64 tensors
/// but fit in i32 range. This reads raw bytes (pairs of u32) and extracts low 32 bits.
///
/// Buffers:
/// - input_i64: i64 data as pairs of u32 (little-endian)
/// - output_i32: output i32 values
pub fn launch_cast_i64_to_i32(
    cache: &PipelineCache,
    queue: &Queue,
    input_i64: &Buffer,
    output_i32: &Buffer,
    count: usize,
) -> Result<()> {
    let shader_source = include_str!("sparse_level_compute.wgsl");

    // Extract just the cast_i64_to_i32 compute function
    let cast_module = cache.get_or_create_module_from_source(
        "cast_i64_to_i32",
        &format!(
            r#"
@group(0) @binding(0) var<storage, read> input_i64: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_i32: array<i32>;

@compute @workgroup_size(256)
fn cast_i64_to_i32(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= arrayLength(&output_i32)) {{ return; }}
    output_i32[idx] = i32(input_i64[2u * idx]);
}}
"#
        ),
    );

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 0,
    });

    let pipeline =
        cache.get_or_create_pipeline("cast_i64_to_i32", "cast_i64_to_i32", &cast_module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input_i64, output_i32]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cast_i64_to_i32"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cast_i64_to_i32"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(count), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Level computation kernels
// ============================================================================

/// Launch iterative lower triangle level computation.
///
/// Iteratively computes level[i] = max(level[j] + 1) for all j < i.
/// Returns immediately; convergence happens on GPU.
///
/// Buffers:
/// - row_ptrs: CSR row pointers (I32)
/// - col_indices: CSR column indices (I32)
/// - levels: Current level values (I32, atomic)
/// - changed: Convergence flag (U32 array[1], atomic)
pub fn launch_compute_levels_lower_iter(
    cache: &PipelineCache,
    queue: &Queue,
    row_ptrs: &Buffer,
    col_indices: &Buffer,
    levels: &Buffer,
    changed: &Buffer,
    n: usize,
) -> Result<()> {
    let shader_source = include_str!("sparse_level_compute.wgsl");

    let module = cache.get_or_create_module_from_source(
        "compute_levels_lower_iter",
        &format!(
            r#"
@group(0) @binding(0) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> levels: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> changed: array<atomic<u32>>;

struct Params {{
    n: u32,
    iteration: u32,
}}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn compute_levels_lower_iter(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.n) {{ return; }}

    var max_level: i32 = -1;
    let row_start = row_ptrs[i];
    let row_end = row_ptrs[i + 1u];

    for (var idx = row_start; idx < row_end; idx = idx + 1) {{
        let j = col_indices[idx];
        if (j < i32(i)) {{
            let j_level = atomicLoad(&levels[u32(j)]);
            if (j_level + 1 > max_level) {{
                max_level = j_level + 1;
            }}
        }}
    }}

    if (max_level > 0) {{
        let old_level = atomicExchange(&levels[i], max_level);
        if (max_level > old_level) {{
            atomicStore(&changed[0], 1u);
        }}
    }}
}}
"#
        ),
    );

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });

    let pipeline = cache.get_or_create_pipeline(
        "compute_levels_lower_iter",
        "compute_levels_lower_iter",
        &module,
        &layout,
    );

    let params_buffer = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("compute_levels_params"),
        size: 8,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params: [u32; 2] = [n as u32, 0];
    queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

    let bind_group = cache.create_bind_group(
        &layout,
        &[row_ptrs, col_indices, levels, changed, &params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compute_levels_lower_iter"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_levels_lower_iter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch iterative upper triangle level computation.
///
/// Iteratively computes level[i] = max(level[j] + 1) for all j > i.
pub fn launch_compute_levels_upper_iter(
    cache: &PipelineCache,
    queue: &Queue,
    row_ptrs: &Buffer,
    col_indices: &Buffer,
    levels: &Buffer,
    changed: &Buffer,
    n: usize,
) -> Result<()> {
    let shader_source = include_str!("sparse_level_compute.wgsl");

    let module = cache.get_or_create_module_from_source(
        "compute_levels_upper_iter",
        &format!(
            r#"
@group(0) @binding(0) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> levels: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> changed: array<atomic<u32>>;

struct Params {{
    n: u32,
    iteration: u32,
}}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn compute_levels_upper_iter(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.n) {{ return; }}

    var max_level: i32 = -1;
    let row_start = row_ptrs[i];
    let row_end = row_ptrs[i + 1u];

    for (var idx = row_start; idx < row_end; idx = idx + 1) {{
        let j = col_indices[idx];
        if (j > i32(i)) {{
            let j_level = atomicLoad(&levels[u32(j)]);
            if (j_level + 1 > max_level) {{
                max_level = j_level + 1;
            }}
        }}
    }}

    if (max_level > 0) {{
        let old_level = atomicExchange(&levels[i], max_level);
        if (max_level > old_level) {{
            atomicStore(&changed[0], 1u);
        }}
    }}
}}
"#
        ),
    );

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });

    let pipeline = cache.get_or_create_pipeline(
        "compute_levels_upper_iter",
        "compute_levels_upper_iter",
        &module,
        &layout,
    );

    let params_buffer = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("compute_levels_params"),
        size: 8,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params: [u32; 2] = [n as u32, 0];
    queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

    let bind_group = cache.create_bind_group(
        &layout,
        &[row_ptrs, col_indices, levels, changed, &params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compute_levels_upper_iter"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_levels_upper_iter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch iterative ILU level computation.
///
/// Iteratively computes level[i] = max(level[j] + 1) for all j < i.
pub fn launch_compute_levels_ilu_iter(
    cache: &PipelineCache,
    queue: &Queue,
    row_ptrs: &Buffer,
    col_indices: &Buffer,
    levels: &Buffer,
    changed: &Buffer,
    n: usize,
) -> Result<()> {
    let shader_source = include_str!("sparse_level_compute.wgsl");

    let module = cache.get_or_create_module_from_source(
        "compute_levels_ilu_iter",
        &format!(
            r#"
@group(0) @binding(0) var<storage, read> row_ptrs: array<i32>;
@group(0) @binding(1) var<storage, read> col_indices: array<i32>;
@group(0) @binding(2) var<storage, read_write> levels: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> changed: array<atomic<u32>>;

struct Params {{
    n: u32,
    iteration: u32,
}}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn compute_levels_ilu_iter(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.n) {{ return; }}

    var max_level: i32 = -1;
    let row_start = row_ptrs[i];
    let row_end = row_ptrs[i + 1u];

    for (var idx = row_start; idx < row_end; idx = idx + 1) {{
        let j = col_indices[idx];
        if (j < i32(i)) {{
            let j_level = atomicLoad(&levels[u32(j)]);
            if (j_level + 1 > max_level) {{
                max_level = j_level + 1;
            }}
        }}
    }}

    if (max_level > 0) {{
        let old_level = atomicExchange(&levels[i], max_level);
        if (max_level > old_level) {{
            atomicStore(&changed[0], 1u);
        }}
    }}
}}
"#
        ),
    );

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });

    let pipeline = cache.get_or_create_pipeline(
        "compute_levels_ilu_iter",
        "compute_levels_ilu_iter",
        &module,
        &layout,
    );

    let params_buffer = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("compute_levels_params"),
        size: 8,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params: [u32; 2] = [n as u32, 0];
    queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

    let bind_group = cache.create_bind_group(
        &layout,
        &[row_ptrs, col_indices, levels, changed, &params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("compute_levels_ilu_iter"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_levels_ilu_iter"),
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
// Scatter by level
// ============================================================================

/// Launch scatter_by_level kernel - distributes row indices into level_rows array.
///
/// Buffers:
/// - levels: level assignments for each row (I32)
/// - level_ptrs: prefix sum of level counts (U32)
/// - level_offsets: atomic counters for scatter positions (I32)
/// - level_rows: output array with rows grouped by level (U32)
pub fn launch_scatter_by_level(
    cache: &PipelineCache,
    queue: &Queue,
    levels: &Buffer,
    level_ptrs: &Buffer,
    level_offsets: &Buffer,
    level_rows: &Buffer,
    num_levels: usize,
    n: usize,
) -> Result<()> {
    let module = cache.get_or_create_module_from_source(
        "scatter_by_level",
        &format!(
            r#"
@group(0) @binding(0) var<storage, read> levels: array<i32>;
@group(0) @binding(1) var<storage, read> level_ptrs: array<u32>;
@group(0) @binding(2) var<storage, read_write> level_offsets: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> level_rows: array<u32>;

struct Params {{
    n: u32,
    num_levels: u32,
}}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn scatter_by_level(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.n) {{ return; }}

    let level = levels[i];
    if (level >= 0 && u32(level) < params.num_levels) {{
        let pos = atomicAdd(&level_offsets[u32(level)], 1u);
        let row_start = level_ptrs[u32(level)];
        level_rows[row_start + pos] = i;
    }}
}}
"#
        ),
    );

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
    });

    let pipeline =
        cache.get_or_create_pipeline("scatter_by_level", "scatter_by_level", &module, &layout);

    let params_buffer = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("scatter_by_level_params"),
        size: 8,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let params: [u32; 2] = [n as u32, num_levels as u32];
    queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            levels,
            level_ptrs,
            level_offsets,
            level_rows,
            &params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scatter_by_level"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter_by_level"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
