//! Walsh-Hadamard transform WGSL kernel launchers, F32 only.
//!
//! - `launch_fwht_local` - one workgroup per tile, strides below the tile width
//! - `launch_fwht_global` - one in-place elementwise dispatch per wide stride

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, WORKGROUP_SIZE};
use crate::dtype::DType;
use crate::error::{Error, Result};

const FWHT_LOCAL_SHADER: &str = include_str!("fwht_local_f32.wgsl");
const FWHT_GLOBAL_SHADER: &str = include_str!("fwht_global_f32.wgsl");

/// The tile-width line the local shader ships with; `launch_fwht_local`
/// rewrites it to the selected width before compiling.
const CHUNK_TEMPLATE_LINE: &str = "const CHUNK: u32 = 4096u;";

/// Largest tile width the local shader accepts: 4096 f32 = 16 KB of
/// workgroup storage, the WebGPU default `maxComputeWorkgroupStorageSize`.
pub const FWHT_MAX_CHUNK: usize = 4096;

/// Smallest tile width the local shader accepts: one element per invocation.
pub const FWHT_MIN_CHUNK: usize = 256;

/// Workgroups per dispatch row. Both shaders fold `workgroup_id.y` back into
/// a linear index with this constant.
const GRID_X: u32 = 65535;

fn check_f32(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Splits `groups` workgroups over x and y so neither exceeds the per-axis
/// dispatch cap.
fn grid_2d(groups: usize, op: &'static str) -> Result<(u32, u32)> {
    let groups = u32::try_from(groups)
        .map_err(|_| Error::backend_limitation("WebGPU", op, "workgroup count exceeds u32"))?;
    if groups <= GRID_X {
        return Ok((groups.max(1), 1));
    }
    let y = groups.div_ceil(GRID_X);
    if y > GRID_X {
        return Err(Error::backend_limitation(
            "WebGPU",
            op,
            format!("{groups} workgroups exceed the dispatch grid"),
        ));
    }
    Ok((GRID_X, y))
}

/// Launch the local pass: signs, scale, and every butterfly stride below
/// `chunk`, one workgroup per `chunk`-wide tile of the flattened buffer.
///
/// `chunk` is a power of two in `FWHT_MIN_CHUNK..=FWHT_MAX_CHUNK`; the caller
/// picks it from the device's workgroup storage limit. `total` is
/// `rows * last_dim`. `signs` is the sign row or the dummy buffer the caller
/// binds when `has_signs` is 0.
#[allow(clippy::too_many_arguments)]
pub fn launch_fwht_local(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    signs: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total: usize,
    chunk: usize,
    dtype: DType,
) -> Result<()> {
    check_f32(dtype, "fwht")?;
    if !chunk.is_power_of_two() || !(FWHT_MIN_CHUNK..=FWHT_MAX_CHUNK).contains(&chunk) {
        return Err(Error::Internal(format!(
            "fwht: tile width {chunk} is not a power of two in {FWHT_MIN_CHUNK}..={FWHT_MAX_CHUNK}"
        )));
    }
    if !FWHT_LOCAL_SHADER.contains(CHUNK_TEMPLATE_LINE) {
        return Err(Error::Internal(format!(
            "fwht: fwht_local_f32.wgsl no longer contains `{CHUNK_TEMPLATE_LINE}`"
        )));
    }

    let module_key = format!("fwht_local_f32_c{chunk}");
    let source = FWHT_LOCAL_SHADER.replace(
        CHUNK_TEMPLATE_LINE,
        &format!("const CHUNK: u32 = {chunk}u;"),
    );
    let module = cache.get_or_create_module_from_source(&module_key, &source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 2,
    });
    let pipeline =
        cache.get_or_create_dynamic_pipeline(&module_key, "fwht_local_f32", &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[input, signs, output, params_buffer]);

    let (gx, gy) = grid_2d(total.div_ceil(chunk), "fwht")?;

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fwht_local"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fwht_local"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch one global-pass stride: the in-place stride-`h` butterfly over
/// every pair of the output buffer.
///
/// `params_buffer` must carry the stride in its `h` field; the caller
/// creates one params buffer per stride. `pairs` is `rows * last_dim / 2`.
pub fn launch_fwht_global(
    cache: &PipelineCache,
    queue: &Queue,
    output: &Buffer,
    params_buffer: &Buffer,
    pairs: usize,
    dtype: DType,
) -> Result<()> {
    check_f32(dtype, "fwht")?;

    let module = cache.get_or_create_module("fwht_global_f32", FWHT_GLOBAL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("fwht_global_f32", "fwht_global_f32", &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[output, params_buffer]);

    let groups = pairs.div_ceil(WORKGROUP_SIZE as usize);
    let (gx, gy) = grid_2d(groups, "fwht")?;

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fwht_global"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fwht_global"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
