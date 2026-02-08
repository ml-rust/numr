//! Statistics WGSL kernel launchers
//!
//! Provides launchers for statistics operations:
//! - Mode (most frequent value along a dimension)
//!
//! All operations run entirely on GPU with no CPU fallback.

use wgpu::{Buffer, Queue};

use super::generator::is_wgpu_supported;
use super::pipeline::{LayoutKey, PipelineCache};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Mode Shader Generation
// ============================================================================

/// Get WGSL type string for dtype
fn wgsl_type_str(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::U32 => "u32",
        _ => "f32", // Fallback, should be validated before calling
    }
}

/// Get suffix for kernel names
fn dtype_suffix_str(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::I32 => "i32",
        DType::U32 => "u32",
        _ => "f32", // Fallback, should be validated before calling
    }
}

/// Generate WGSL shader for mode operation
fn generate_mode_shader(dtype: DType) -> String {
    let wgsl_t = wgsl_type_str(dtype);
    let suffix = dtype_suffix_str(dtype);

    format!(
        r#"
// Mode shader for {wgsl_t}
// Finds most frequent value in sorted data along reduce dimension

struct ModeParams {{
    outer_size: u32,
    reduce_size: u32,
    inner_size: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read_write> sorted: array<{wgsl_t}>;
@group(0) @binding(1) var<storage, read_write> mode_values: array<{wgsl_t}>;
@group(0) @binding(2) var<storage, read_write> mode_counts: array<i32>;
@group(0) @binding(3) var<uniform> params: ModeParams;

@compute @workgroup_size(1)
fn mode_dim_{suffix}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let out_idx = gid.x;
    let total_outputs = params.outer_size * params.inner_size;

    if (out_idx >= total_outputs) {{
        return;
    }}

    let outer = out_idx / params.inner_size;
    let inner = out_idx % params.inner_size;
    let base = outer * params.reduce_size * params.inner_size + inner;

    if (params.reduce_size == 0u) {{
        return;
    }}

    // Initialize with first element
    var best_val = sorted[base];
    var best_count: i32 = 1;
    var curr_val = best_val;
    var curr_count: i32 = 1;

    // Scan through sorted slice
    for (var r: u32 = 1u; r < params.reduce_size; r = r + 1u) {{
        let idx = base + r * params.inner_size;
        let val = sorted[idx];

        if (val == curr_val) {{
            curr_count = curr_count + 1;
        }} else {{
            if (curr_count > best_count) {{
                best_val = curr_val;
                best_count = curr_count;
            }}
            curr_val = val;
            curr_count = 1;
        }}
    }}

    // Check final run
    if (curr_count > best_count) {{
        best_val = curr_val;
        best_count = curr_count;
    }}

    mode_values[out_idx] = best_val;
    mode_counts[out_idx] = best_count;
}}
"#,
        wgsl_t = wgsl_t,
        suffix = suffix
    )
}

/// Get module key for caching
fn mode_module_key(dtype: DType) -> String {
    format!("mode_{}", dtype_suffix_str(dtype))
}

// ============================================================================
// Launcher Functions
// ============================================================================

/// Launch mode operation kernel along a dimension.
///
/// Input must be pre-sorted along the reduce dimension.
/// Supports F32, I32, U32 dtypes.
///
/// Parameters:
/// - sorted: Pre-sorted input buffer
/// - mode_values: Output buffer for mode values
/// - mode_counts: Output buffer for mode counts (i32)
/// - params_buffer: Uniform buffer with (outer_size, reduce_size, inner_size, pad)
/// - num_outputs: outer_size * inner_size
pub fn launch_mode_dim(
    cache: &PipelineCache,
    queue: &Queue,
    sorted: &Buffer,
    mode_values: &Buffer,
    mode_counts: &Buffer,
    params_buffer: &Buffer,
    num_outputs: usize,
    dtype: DType,
) -> Result<()> {
    if !is_wgpu_supported(dtype) {
        return Err(Error::UnsupportedDType { dtype, op: "mode" });
    }

    let suffix = dtype_suffix_str(dtype);
    let entry_point = format!("mode_dim_{}", suffix);
    // Leak entry_point to get static reference (cached, so leak is acceptable)
    let static_entry_point: &'static str = Box::leak(entry_point.into_boxed_str());

    // Generate shader and module key
    let shader = generate_mode_shader(dtype);
    let module_key = mode_module_key(dtype);
    let static_module_key: &'static str = Box::leak(module_key.into_boxed_str());
    let static_shader: &'static str = Box::leak(shader.into_boxed_str());

    // Get or create shader module
    let module = cache.get_or_create_module(static_module_key, static_shader);

    // Layout: 3 storage buffers + 1 uniform buffer
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    // Get or create pipeline
    let pipeline =
        cache.get_or_create_pipeline(static_module_key, static_entry_point, &module, &layout);

    // Create bind group
    let bind_group =
        cache.create_bind_group(&layout, &[sorted, mode_values, mode_counts, params_buffer]);

    // Create command encoder and dispatch
    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mode_dim"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mode_dim"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per output element
        pass.dispatch_workgroups(num_outputs as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch full mode operation (reduce entire tensor to single value).
#[allow(dead_code)] // May be used in future for full tensor mode
pub fn launch_mode_full(
    cache: &PipelineCache,
    queue: &Queue,
    sorted: &Buffer,
    mode_value: &Buffer,
    mode_count: &Buffer,
    numel_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    if !is_wgpu_supported(dtype) {
        return Err(Error::UnsupportedDType { dtype, op: "mode" });
    }

    let suffix = dtype_suffix_str(dtype);
    let entry_point = format!("mode_full_{}", suffix);
    let static_entry_point: &'static str = Box::leak(entry_point.into_boxed_str());

    let shader = generate_mode_shader(dtype);
    let module_key = format!("mode_full_{}", suffix);
    let static_module_key: &'static str = Box::leak(module_key.into_boxed_str());
    let static_shader: &'static str = Box::leak(shader.into_boxed_str());

    let module = cache.get_or_create_module(static_module_key, static_shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline(static_module_key, static_entry_point, &module, &layout);
    let bind_group =
        cache.create_bind_group(&layout, &[sorted, mode_value, mode_count, numel_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mode_full"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mode_full"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
