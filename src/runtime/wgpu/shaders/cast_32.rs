//! Launcher for the casts between the 32-bit dtypes (F32, I32, U32).

use wgpu::{Buffer, BufferBinding, Queue};

use super::linear_grid::dispatch_linear;
use super::pipeline::{LayoutKey, PipelineCache};
use crate::dtype::DType;
use crate::error::{Error, Result};

const CAST_F32_TO_I32_SHADER: &str = include_str!("cast_f32_to_i32.wgsl");
const CAST_F32_TO_U32_SHADER: &str = include_str!("cast_f32_to_u32.wgsl");
const CAST_I32_TO_F32_SHADER: &str = include_str!("cast_i32_to_f32.wgsl");
const CAST_I32_TO_U32_SHADER: &str = include_str!("cast_i32_to_u32.wgsl");
const CAST_U32_TO_F32_SHADER: &str = include_str!("cast_u32_to_f32.wgsl");
const CAST_U32_TO_I32_SHADER: &str = include_str!("cast_u32_to_i32.wgsl");

/// Launch a cast operation: `out[i] = DstType(a[i])`. Supports F32 ↔ I32 ↔ U32.
///
/// `a` and `out` are the bound windows of the source and destination.
/// `params_buffer` holds a `CastParams { numel }` for those windows.
#[allow(clippy::too_many_arguments)]
pub fn launch_cast_op(
    cache: &PipelineCache,
    queue: &Queue,
    a: BufferBinding<'_>,
    out: BufferBinding<'_>,
    params_buffer: &Buffer,
    numel: usize,
    src_dtype: DType,
    dst_dtype: DType,
) -> Result<()> {
    if src_dtype == dst_dtype {
        return Ok(());
    }

    let (module_name, entry_point, shader_source): (&'static str, &'static str, &'static str) =
        match (src_dtype, dst_dtype) {
            (DType::F32, DType::I32) => ("cast_f32_i32", "cast_f32_to_i32", CAST_F32_TO_I32_SHADER),
            (DType::F32, DType::U32) => ("cast_f32_u32", "cast_f32_to_u32", CAST_F32_TO_U32_SHADER),
            (DType::I32, DType::F32) => ("cast_i32_f32", "cast_i32_to_f32", CAST_I32_TO_F32_SHADER),
            (DType::I32, DType::U32) => ("cast_i32_u32", "cast_i32_to_u32", CAST_I32_TO_U32_SHADER),
            (DType::U32, DType::F32) => ("cast_u32_f32", "cast_u32_to_f32", CAST_U32_TO_F32_SHADER),
            (DType::U32, DType::I32) => ("cast_u32_i32", "cast_u32_to_i32", CAST_U32_TO_I32_SHADER),
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype: src_dtype,
                    op: "cast",
                });
            }
        };

    let module = cache.get_or_create_module(module_name, shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_name, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group_from_bindings(
        &layout,
        &[a, out, params_buffer.as_entire_buffer_binding()],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cast"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cast"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        dispatch_linear(&mut pass, numel);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
