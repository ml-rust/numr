//! WGSL shader launchers for 2:4 structured sparsity operations

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::error::Result;

const PRUNE_SHADER: &str = include_str!("sparse_24_prune.wgsl");
const DECOMPRESS_SHADER: &str = include_str!("sparse_24_decompress.wgsl");

/// Parameters for 2:4 sparse operations (matches WGSL Params struct)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sparse24Params {
    /// Total number of 2:4 groups across all rows (m * num_groups_per_row).
    pub total_groups: u32,
    /// Number of groups per row (k / 4).
    pub num_groups_per_row: u32,
    /// Number of metadata columns per row.
    pub meta_cols: u32,
    /// Half of the K dimension (k / 2), i.e. number of non-zero values per row.
    pub half_k: u32,
    /// Full K dimension of the dense matrix.
    pub k: u32,
    /// Padding to satisfy WGSL 16-byte uniform alignment.
    pub _pad0: u32,
    /// Padding to satisfy WGSL 16-byte uniform alignment.
    pub _pad1: u32,
    /// Padding to satisfy WGSL 16-byte uniform alignment.
    pub _pad2: u32,
}

/// Launch prune-to-2:4 shader.
pub fn launch_sparse_24_prune(
    cache: &PipelineCache,
    queue: &Queue,
    dense: &Buffer,
    compressed: &Buffer,
    metadata: &Buffer,
    params_buffer: &Buffer,
    total_groups: usize,
) -> Result<()> {
    let module = cache.get_or_create_module("sparse_24_prune", PRUNE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 1,
    });
    let pipeline = cache.get_or_create_dynamic_pipeline(
        "sparse_24_prune",
        "sparse_24_prune_f32",
        &module,
        &layout,
    );
    let bind_group =
        cache.create_bind_group(&layout, &[dense, compressed, metadata, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sparse_24_prune"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sparse_24_prune"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_groups), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch decompress-from-2:4 shader.
pub fn launch_sparse_24_decompress(
    cache: &PipelineCache,
    queue: &Queue,
    compressed: &Buffer,
    metadata: &Buffer,
    dense: &Buffer,
    params_buffer: &Buffer,
    total_groups: usize,
) -> Result<()> {
    let module = cache.get_or_create_module("sparse_24_decompress", DECOMPRESS_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 2,
    });
    let pipeline = cache.get_or_create_dynamic_pipeline(
        "sparse_24_decompress",
        "sparse_24_decompress_f32",
        &module,
        &layout,
    );
    let bind_group =
        cache.create_bind_group(&layout, &[compressed, metadata, dense, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sparse_24_decompress"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sparse_24_decompress"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_groups), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
