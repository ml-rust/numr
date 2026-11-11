//! Integer-only binary32 stepping with checked strided addressing.

use wgpu::Buffer;

use super::linear_grid::{element_count_u32, linear_grid};
use super::pipeline::LayoutKey;
use crate::error::{Error, Result};
use crate::runtime::wgpu::client::get_buffer;
use crate::runtime::wgpu::ops::helpers::create_storage_buffer;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

const SOURCE: &str = include_str!("next_representable_f32.wgsl");

fn limitation(op: &'static str, reason: impl Into<String>) -> Error {
    Error::backend_limitation("WebGPU", op, reason.into())
}

fn address(value: usize, name: &str, op: &'static str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| limitation(op, format!("{name} {value} exceeds u32 addressing")))
}

fn parameters(input: &Tensor<WgpuRuntime>, upward: bool, op: &'static str) -> Result<Vec<u32>> {
    let rank = input.ndim();
    let words = rank
        .checked_mul(2)
        .and_then(|n| n.checked_add(4))
        .ok_or_else(|| limitation(op, "layout parameter count overflows usize"))?;
    address(words, "layout parameter count", op)?;
    if input.strides().len() != rank {
        return Err(limitation(op, "shape and stride ranks differ"));
    }
    let mut params = Vec::with_capacity(words);
    params.extend_from_slice(&[
        element_count_u32(input.numel(), op)?,
        address(rank, "rank", op)?,
        address(input.offset(), "input offset", op)?,
        u32::from(upward),
    ]);
    for &extent in input.shape() {
        if extent == 0 {
            return Err(limitation(
                op,
                "nonempty dispatch contains a zero dimension",
            ));
        }
        params.push(address(extent, "dimension", op)?);
    }
    let mut low = input.offset() as i128;
    let mut high = low;
    for (&extent, &stride) in input.shape().iter().zip(input.strides()) {
        let stride = i32::try_from(stride)
            .map_err(|_| limitation(op, format!("stride {stride} exceeds i32 addressing")))?;
        params.push(stride as u32);
        let span = (extent - 1) as i128 * i128::from(stride);
        low = low
            .checked_add(span.min(0))
            .ok_or_else(|| limitation(op, "minimum input address overflows i128"))?;
        high = high
            .checked_add(span.max(0))
            .ok_or_else(|| limitation(op, "maximum input address overflows i128"))?;
    }
    address(input.storage().len(), "input storage length", op)?;
    if low < 0 || high >= input.storage().len() as i128 || high > i128::from(u32::MAX) {
        return Err(limitation(
            op,
            format!("input address range {low}..={high} exceeds storage or u32 addressing"),
        ));
    }
    Ok(params)
}

/// Dispatch raw integer loads and stores without changing floating arithmetic contracts.
pub(crate) fn launch_next_representable(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    output: &Buffer,
    upward: bool,
    op: &'static str,
) -> Result<()> {
    let params = parameters(input, upward, op)?;
    let input_buffer = get_buffer(input.storage().ptr())
        .ok_or_else(|| Error::Internal(format!("{op}: input buffer is not in the registry")))?;
    let parameter_bytes = (params.len() as u64)
        .checked_mul(4)
        .ok_or_else(|| limitation(op, "layout parameter size overflows u64"))?;
    let limits = client.wgpu_device().limits();
    for size in [input_buffer.size(), output.size(), parameter_bytes] {
        if size > limits.max_storage_buffer_binding_size || size > limits.max_buffer_size {
            return Err(limitation(
                op,
                format!("buffer size {size} exceeds WebGPU storage binding limits"),
            ));
        }
    }
    let params_buffer = create_storage_buffer(client, &params);
    let cache = client.pipeline_cache();
    let module = cache.get_or_create_module("next_representable_f32", SOURCE);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 0,
        num_readonly_storage: 2,
    });
    let pipeline = cache.get_or_create_pipeline(
        "next_representable_f32",
        "next_representable_f32",
        &module,
        &layout,
    );
    let bindings = cache.create_bind_group(&layout, &[&input_buffer, &params_buffer, output]);
    let mut encoder = client
        .wgpu_device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(op) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bindings), &[]);
        let (x, y) = linear_grid(input.numel());
        pass.dispatch_workgroups(x, y, 1);
    }
    client
        .wgpu_queue()
        .submit(std::iter::once(encoder.finish()));
    Ok(())
}
