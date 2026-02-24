//! Where (conditional select) WGSL kernel launchers. F32/I32/U32 supported.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Static shaders — element-wise (4 storage + 1 uniform)
// ============================================================================

const WHERE_COND_F32_F32: &str = include_str!("where_cond_f32_f32.wgsl");
const WHERE_COND_F32_I32: &str = include_str!("where_cond_f32_i32.wgsl");
const WHERE_COND_F32_U32: &str = include_str!("where_cond_f32_u32.wgsl");
const WHERE_COND_I32_F32: &str = include_str!("where_cond_i32_f32.wgsl");
const WHERE_COND_I32_I32: &str = include_str!("where_cond_i32_i32.wgsl");
const WHERE_COND_I32_U32: &str = include_str!("where_cond_i32_u32.wgsl");
const WHERE_COND_U32_F32: &str = include_str!("where_cond_u32_f32.wgsl");
const WHERE_COND_U32_I32: &str = include_str!("where_cond_u32_i32.wgsl");
const WHERE_COND_U32_U32: &str = include_str!("where_cond_u32_u32.wgsl");

// ============================================================================
// Static shaders — broadcast (8 storage + 1 uniform)
// ============================================================================

const WHERE_BC_F32_F32: &str = include_str!("where_broadcast_cond_f32_f32.wgsl");
const WHERE_BC_F32_I32: &str = include_str!("where_broadcast_cond_f32_i32.wgsl");
const WHERE_BC_F32_U32: &str = include_str!("where_broadcast_cond_f32_u32.wgsl");
const WHERE_BC_I32_F32: &str = include_str!("where_broadcast_cond_i32_f32.wgsl");
const WHERE_BC_I32_I32: &str = include_str!("where_broadcast_cond_i32_i32.wgsl");
const WHERE_BC_I32_U32: &str = include_str!("where_broadcast_cond_i32_u32.wgsl");
const WHERE_BC_U32_F32: &str = include_str!("where_broadcast_cond_u32_f32.wgsl");
const WHERE_BC_U32_I32: &str = include_str!("where_broadcast_cond_u32_i32.wgsl");
const WHERE_BC_U32_U32: &str = include_str!("where_broadcast_cond_u32_u32.wgsl");

// ============================================================================
// Shader dispatch helpers
// ============================================================================

/// Returns (shader, module_key, entry_point) for element-wise where_cond.
fn where_shader_info(
    cond_dtype: DType,
    out_dtype: DType,
) -> Result<(&'static str, &'static str, &'static str)> {
    Ok(match (cond_dtype, out_dtype) {
        (DType::F32, DType::F32) => (
            WHERE_COND_F32_F32,
            "where_cond_f32_f32",
            "where_cond_f32_f32",
        ),
        (DType::F32, DType::I32) => (
            WHERE_COND_F32_I32,
            "where_cond_f32_i32",
            "where_cond_f32_i32",
        ),
        (DType::F32, DType::U32) => (
            WHERE_COND_F32_U32,
            "where_cond_f32_u32",
            "where_cond_f32_u32",
        ),
        (DType::I32, DType::F32) => (
            WHERE_COND_I32_F32,
            "where_cond_i32_f32",
            "where_cond_i32_f32",
        ),
        (DType::I32, DType::I32) => (
            WHERE_COND_I32_I32,
            "where_cond_i32_i32",
            "where_cond_i32_i32",
        ),
        (DType::I32, DType::U32) => (
            WHERE_COND_I32_U32,
            "where_cond_i32_u32",
            "where_cond_i32_u32",
        ),
        (DType::U32, DType::F32) => (
            WHERE_COND_U32_F32,
            "where_cond_u32_f32",
            "where_cond_u32_f32",
        ),
        (DType::U32, DType::I32) => (
            WHERE_COND_U32_I32,
            "where_cond_u32_i32",
            "where_cond_u32_i32",
        ),
        (DType::U32, DType::U32) => (
            WHERE_COND_U32_U32,
            "where_cond_u32_u32",
            "where_cond_u32_u32",
        ),
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: cond_dtype,
                op: "where_cond (WebGPU)",
            });
        }
    })
}

/// Returns (shader, module_key, entry_point) for broadcast where_cond.
fn where_broadcast_shader_info(
    cond_dtype: DType,
    out_dtype: DType,
) -> Result<(&'static str, &'static str, &'static str)> {
    Ok(match (cond_dtype, out_dtype) {
        (DType::F32, DType::F32) => (
            WHERE_BC_F32_F32,
            "where_broadcast_cond_f32_f32",
            "where_broadcast_cond_f32_f32",
        ),
        (DType::F32, DType::I32) => (
            WHERE_BC_F32_I32,
            "where_broadcast_cond_f32_i32",
            "where_broadcast_cond_f32_i32",
        ),
        (DType::F32, DType::U32) => (
            WHERE_BC_F32_U32,
            "where_broadcast_cond_f32_u32",
            "where_broadcast_cond_f32_u32",
        ),
        (DType::I32, DType::F32) => (
            WHERE_BC_I32_F32,
            "where_broadcast_cond_i32_f32",
            "where_broadcast_cond_i32_f32",
        ),
        (DType::I32, DType::I32) => (
            WHERE_BC_I32_I32,
            "where_broadcast_cond_i32_i32",
            "where_broadcast_cond_i32_i32",
        ),
        (DType::I32, DType::U32) => (
            WHERE_BC_I32_U32,
            "where_broadcast_cond_i32_u32",
            "where_broadcast_cond_i32_u32",
        ),
        (DType::U32, DType::F32) => (
            WHERE_BC_U32_F32,
            "where_broadcast_cond_u32_f32",
            "where_broadcast_cond_u32_f32",
        ),
        (DType::U32, DType::I32) => (
            WHERE_BC_U32_I32,
            "where_broadcast_cond_u32_i32",
            "where_broadcast_cond_u32_i32",
        ),
        (DType::U32, DType::U32) => (
            WHERE_BC_U32_U32,
            "where_broadcast_cond_u32_u32",
            "where_broadcast_cond_u32_u32",
        ),
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: cond_dtype,
                op: "where_broadcast_cond (WebGPU)",
            });
        }
    })
}

// ============================================================================
// Kernel Launchers
// ============================================================================

/// Launch where conditional operation kernel (F32-only legacy wrapper).
///
/// Computes `out[i] = cond[i] != 0 ? x[i] : y[i]` for all elements.
/// Delegates to `launch_where_generic_op` with F32 condition dtype.
#[allow(clippy::too_many_arguments)]
pub fn launch_where_op(
    cache: &PipelineCache,
    queue: &Queue,
    cond: &Buffer,
    x: &Buffer,
    y: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    launch_where_generic_op(
        cache,
        queue,
        cond,
        x,
        y,
        out,
        params_buffer,
        numel,
        DType::F32,
        dtype,
    )
}

/// Launch where conditional operation kernel with generic condition dtype.
///
/// Computes `out[i] = cond[i] != 0 ? x[i] : y[i]` for all elements.
/// Supports F32, I32, U32 condition dtypes.
#[allow(clippy::too_many_arguments)]
pub fn launch_where_generic_op(
    cache: &PipelineCache,
    queue: &Queue,
    cond: &Buffer,
    x: &Buffer,
    y: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    cond_dtype: DType,
    out_dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = where_shader_info(cond_dtype, out_dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[cond, x, y, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("where_cond"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("where_cond"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch broadcast where conditional operation kernel.
///
/// Computes `out[i] = cond[cond_offset] != 0 ? x[x_offset] : y[y_offset]`
/// with broadcasting support via per-dimension stride buffers.
#[allow(clippy::too_many_arguments)]
pub fn launch_where_broadcast_op(
    cache: &PipelineCache,
    queue: &Queue,
    cond: &Buffer,
    x: &Buffer,
    y: &Buffer,
    out: &Buffer,
    cond_strides: &Buffer,
    x_strides: &Buffer,
    y_strides: &Buffer,
    out_shape: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    cond_dtype: DType,
    out_dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = where_broadcast_shader_info(cond_dtype, out_dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 8,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            cond,
            x,
            y,
            out,
            cond_strides,
            x_strides,
            y_strides,
            out_shape,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("where_broadcast"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("where_broadcast"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
