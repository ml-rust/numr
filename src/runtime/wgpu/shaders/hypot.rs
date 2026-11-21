//! F32 magnitude dispatch with integer storage and strided addressing.

use wgpu::{Buffer, BufferUsages};

use super::linear_grid::{element_count_u32, linear_grid};
use super::pipeline::LayoutKey;
use crate::error::{Error, Result};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

const SOURCE: &str = include_str!("hypot_f32.wgsl");

fn address(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::backend_limitation(
            "WebGPU",
            "hypot",
            format!("{name} {value} exceeds u32 addressing"),
        )
    })
}

/// Launch one invocation per output element, reading broadcast views directly.
pub(crate) fn launch_hypot(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    output: &Buffer,
) -> Result<()> {
    let count = a.numel();
    let rank = a.ndim();
    let mut params = Vec::with_capacity(4 + 3 * rank);
    params.extend_from_slice(&[
        element_count_u32(count, "hypot")?,
        address(rank, "rank")?,
        address(a.offset(), "first input offset")?,
        address(b.offset(), "second input offset")?,
    ]);
    for &extent in a.shape() {
        params.push(address(extent, "dimension")?);
    }
    for input in [a, b] {
        for &stride in input.strides() {
            let stride = i32::try_from(stride).map_err(|_| {
                Error::backend_limitation(
                    "WebGPU",
                    "hypot",
                    format!("stride {stride} exceeds i32 addressing"),
                )
            })?;
            params.push(stride as u32);
        }
        address(input.storage().len(), "input storage length")?;
    }
    let params_buffer = client.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("hypot_params"),
        size: (params.len() * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client.write_buffer(&params_buffer, &params);
    let input_buffer = |input: &Tensor<WgpuRuntime>| {
        crate::runtime::wgpu::client::get_buffer(input.storage().ptr())
            .ok_or_else(|| Error::Internal("hypot: input buffer is not in the registry".into()))
    };
    let a_buffer = input_buffer(a)?;
    let b_buffer = input_buffer(b)?;
    let cache = client.pipeline_cache();
    let module = cache.get_or_create_module("hypot_f32", SOURCE);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 0,
        num_readonly_storage: 3,
    });
    let pipeline = cache.get_or_create_pipeline("hypot_f32", "hypot_f32", &module, &layout);
    let bindings =
        cache.create_bind_group(&layout, &[&a_buffer, &b_buffer, &params_buffer, output])?;
    let mut encoder =
        client
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hypot"),
            });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hypot"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bindings), &[]);
        let (x, y) = linear_grid(count);
        pass.dispatch_workgroups(x, y, 1);
    }
    client
        .wgpu_queue()
        .submit(std::iter::once(encoder.finish()));
    Ok(())
}
