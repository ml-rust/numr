//! Semiring matrix multiplication WGSL kernel launchers

use wgpu::{Buffer, Queue};

use super::generator::semiring_matmul::generate_semiring_matmul_shader;
use super::pipeline::{LayoutKey, PipelineCache};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::semiring::SemiringOp;

const TILE_SIZE: u32 = 16;

/// Returns (module_key, entry_point, batched_entry_point) as &'static str.
/// The pipeline cache requires 'static lifetimes for keys.
fn semiring_keys(
    op: SemiringOp,
    dtype: DType,
) -> Result<(&'static str, &'static str, &'static str)> {
    use DType::*;
    use SemiringOp::*;
    match (op, dtype) {
        (MinPlus, F32) => Ok((
            "sr_min_plus_f32",
            "semiring_matmul_min_plus_f32",
            "batched_semiring_matmul_min_plus_f32",
        )),
        (MaxPlus, F32) => Ok((
            "sr_max_plus_f32",
            "semiring_matmul_max_plus_f32",
            "batched_semiring_matmul_max_plus_f32",
        )),
        (MaxMin, F32) => Ok((
            "sr_max_min_f32",
            "semiring_matmul_max_min_f32",
            "batched_semiring_matmul_max_min_f32",
        )),
        (MinMax, F32) => Ok((
            "sr_min_max_f32",
            "semiring_matmul_min_max_f32",
            "batched_semiring_matmul_min_max_f32",
        )),
        (OrAnd, F32) => Ok((
            "sr_or_and_f32",
            "semiring_matmul_or_and_f32",
            "batched_semiring_matmul_or_and_f32",
        )),
        (PlusMax, F32) => Ok((
            "sr_plus_max_f32",
            "semiring_matmul_plus_max_f32",
            "batched_semiring_matmul_plus_max_f32",
        )),

        (MinPlus, I32) => Ok((
            "sr_min_plus_i32",
            "semiring_matmul_min_plus_i32",
            "batched_semiring_matmul_min_plus_i32",
        )),
        (MaxPlus, I32) => Ok((
            "sr_max_plus_i32",
            "semiring_matmul_max_plus_i32",
            "batched_semiring_matmul_max_plus_i32",
        )),
        (MaxMin, I32) => Ok((
            "sr_max_min_i32",
            "semiring_matmul_max_min_i32",
            "batched_semiring_matmul_max_min_i32",
        )),
        (MinMax, I32) => Ok((
            "sr_min_max_i32",
            "semiring_matmul_min_max_i32",
            "batched_semiring_matmul_min_max_i32",
        )),
        (OrAnd, I32) => Ok((
            "sr_or_and_i32",
            "semiring_matmul_or_and_i32",
            "batched_semiring_matmul_or_and_i32",
        )),
        (PlusMax, I32) => Ok((
            "sr_plus_max_i32",
            "semiring_matmul_plus_max_i32",
            "batched_semiring_matmul_plus_max_i32",
        )),

        (MinPlus, U32) => Ok((
            "sr_min_plus_u32",
            "semiring_matmul_min_plus_u32",
            "batched_semiring_matmul_min_plus_u32",
        )),
        (MaxPlus, U32) => Ok((
            "sr_max_plus_u32",
            "semiring_matmul_max_plus_u32",
            "batched_semiring_matmul_max_plus_u32",
        )),
        (MaxMin, U32) => Ok((
            "sr_max_min_u32",
            "semiring_matmul_max_min_u32",
            "batched_semiring_matmul_max_min_u32",
        )),
        (MinMax, U32) => Ok((
            "sr_min_max_u32",
            "semiring_matmul_min_max_u32",
            "batched_semiring_matmul_min_max_u32",
        )),
        (OrAnd, U32) => Ok((
            "sr_or_and_u32",
            "semiring_matmul_or_and_u32",
            "batched_semiring_matmul_or_and_u32",
        )),
        (PlusMax, U32) => Ok((
            "sr_plus_max_u32",
            "semiring_matmul_plus_max_u32",
            "batched_semiring_matmul_plus_max_u32",
        )),

        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "semiring_matmul (WebGPU)",
        }),
    }
}

/// Launch semiring matrix multiplication kernel.
pub fn launch_semiring_matmul(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    op: SemiringOp,
    dtype: DType,
) -> Result<()> {
    let (module_key, entry_point, _) = semiring_keys(op, dtype)?;
    let shader_source = generate_semiring_matmul_shader(dtype, op)?;

    let module = cache.get_or_create_module(module_key, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("semiring_matmul"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("semiring_matmul"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        let num_groups_x = (n as u32 + TILE_SIZE - 1) / TILE_SIZE;
        let num_groups_y = (m as u32 + TILE_SIZE - 1) / TILE_SIZE;
        pass.dispatch_workgroups(num_groups_x, num_groups_y, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch batched semiring matrix multiplication kernel.
pub fn launch_batched_semiring_matmul(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    batch_size: usize,
    op: SemiringOp,
    dtype: DType,
) -> Result<()> {
    let (module_key, _, batched_entry_point) = semiring_keys(op, dtype)?;
    let shader_source = generate_semiring_matmul_shader(dtype, op)?;

    let module = cache.get_or_create_module(module_key, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, batched_entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batched_semiring_matmul"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("batched_semiring_matmul"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        let num_groups_x = (n as u32 + TILE_SIZE - 1) / TILE_SIZE;
        let num_groups_y = (m as u32 + TILE_SIZE - 1) / TILE_SIZE;
        pass.dispatch_workgroups(num_groups_x, num_groups_y, batch_size as u32);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
