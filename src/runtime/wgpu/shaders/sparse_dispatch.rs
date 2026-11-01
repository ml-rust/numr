//! Shared dispatch for sparse WGSL kernels that bind N storage buffers and
//! one uniform buffer.
//!
//! The pipeline layout puts the first `num_readonly` storage buffers in
//! read-only slots and the rest in read-write slots. `num_readonly` must match
//! the `var<storage, read>` count of the entry point, or wgpu refuses the
//! pipeline.

use wgpu::{Buffer, Queue};

use super::linear_grid::dispatch_linear;
use super::pipeline::{LayoutKey, PipelineCache};
use crate::error::Result;

/// One WGSL entry point and its storage-buffer access pattern.
pub(super) struct SparseKernel {
    /// Module cache key.
    pub module: &'static str,
    /// WGSL source of the module.
    pub source: &'static str,
    /// Entry point name.
    pub entry: &'static str,
    /// Leading storage bindings that the entry point declares `read`.
    pub num_readonly: u32,
}

/// Bind `storage` (in binding order) and `uniform` (last binding), then
/// dispatch `threads` invocations over the linear 2-D grid.
pub(super) fn run_sparse_kernel(
    cache: &PipelineCache,
    queue: &Queue,
    kernel: &SparseKernel,
    storage: &[&Buffer],
    uniform: &Buffer,
    threads: usize,
) -> Result<()> {
    let module = cache.get_or_create_module(kernel.module, kernel.source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: storage.len() as u32,
        num_uniform_buffers: 1,
        num_readonly_storage: kernel.num_readonly,
    });
    let pipeline = cache.get_or_create_pipeline(kernel.module, kernel.entry, &module, &layout);

    let mut buffers: Vec<&Buffer> = Vec::with_capacity(storage.len() + 1);
    buffers.extend_from_slice(storage);
    buffers.push(uniform);
    let bind_group = cache.create_bind_group(&layout, &buffers);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(kernel.entry),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(kernel.entry),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        dispatch_linear(&mut pass, threads);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
