//! Statistics WGSL kernel launchers
//!
//! Provides launchers for statistics operations:
//! - Mode (most frequent value along a dimension)
//!
//! All operations run entirely on GPU with no CPU fallback.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Static shaders
// ============================================================================

const MODE_F32: &str = include_str!("statistics_f32.wgsl");
const MODE_I32: &str = include_str!("statistics_i32.wgsl");
const MODE_U32: &str = include_str!("statistics_u32.wgsl");

// ============================================================================
// Shader dispatch helper
// ============================================================================

fn mode_shader_info(dtype: DType) -> Result<(&'static str, &'static str, &'static str)> {
    Ok(match dtype {
        DType::F32 => (MODE_F32, "statistics_f32", "mode_dim_f32"),
        DType::I32 => (MODE_I32, "statistics_i32", "mode_dim_i32"),
        DType::U32 => (MODE_U32, "statistics_u32", "mode_dim_u32"),
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "mode (WebGPU)",
            });
        }
    })
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
    let (shader, module_key, entry_point) = mode_shader_info(dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);
    let bind_group =
        cache.create_bind_group(&layout, &[sorted, mode_values, mode_counts, params_buffer]);

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
        pass.dispatch_workgroups(num_outputs as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch full mode operation (reduce entire tensor to single value).
#[allow(dead_code)]
pub fn launch_mode_full(
    cache: &PipelineCache,
    queue: &Queue,
    sorted: &Buffer,
    mode_value: &Buffer,
    mode_count: &Buffer,
    numel_buffer: &Buffer,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = mode_shader_info(dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);
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
