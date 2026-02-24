//! Distance computation WGSL kernel launchers.
//!
//! Provides launchers for pairwise distance computation using various metrics.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::DistanceMetric;

// Static WGSL shader code
const CDIST_F32: &str = include_str!("distance_cdist_f32.wgsl");
const PDIST_F32: &str = include_str!("distance_pdist_f32.wgsl");
const SQUAREFORM_F32: &str = include_str!("distance_squareform_f32.wgsl");
const SQUAREFORM_INVERSE_F32: &str = include_str!("distance_squareform_inverse_f32.wgsl");

fn check_float_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Convert DistanceMetric to shader index
pub fn metric_to_index(metric: DistanceMetric) -> u32 {
    match metric {
        DistanceMetric::Euclidean => 0,
        DistanceMetric::SquaredEuclidean => 1,
        DistanceMetric::Manhattan => 2,
        DistanceMetric::Chebyshev => 3,
        DistanceMetric::Minkowski(_) => 4,
        DistanceMetric::Cosine => 5,
        DistanceMetric::Correlation => 6,
        DistanceMetric::Hamming => 7,
        DistanceMetric::Jaccard => 8,
    }
}

/// Get Minkowski p value from metric
pub fn metric_p_value(metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Minkowski(p) => p as f32,
        _ => 2.0,
    }
}

/// Launch cdist kernel - pairwise distances between two point sets.
pub fn launch_cdist(
    cache: &PipelineCache,
    queue: &Queue,
    x: &Buffer,
    y: &Buffer,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "cdist")?;

    let name = "cdist_f32";
    let module = cache.get_or_create_module(name, CDIST_F32);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, "main", &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[x, y, out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cdist"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cdist"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch pdist kernel - pairwise distances within one point set (condensed).
pub fn launch_pdist(
    cache: &PipelineCache,
    queue: &Queue,
    x: &Buffer,
    out: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "pdist")?;

    let name = "pdist_f32";
    let module = cache.get_or_create_module(name, PDIST_F32);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, "main", &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[x, out, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pdist"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pdist"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch squareform kernel - condensed to square.
pub fn launch_squareform(
    cache: &PipelineCache,
    queue: &Queue,
    condensed: &Buffer,
    square: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "squareform")?;

    let name = "squareform_f32";
    let module = cache.get_or_create_module(name, SQUAREFORM_F32);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, "main", &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[condensed, square, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("squareform"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("squareform"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch squareform_inverse kernel - square to condensed.
pub fn launch_squareform_inverse(
    cache: &PipelineCache,
    queue: &Queue,
    square: &Buffer,
    condensed: &Buffer,
    params: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }
    check_float_dtype(dtype, "squareform_inverse")?;

    let name = "squareform_inverse_f32";
    let module = cache.get_or_create_module(name, SQUAREFORM_INVERSE_F32);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, "main", &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[square, condensed, params]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("squareform_inverse"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("squareform_inverse"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// Export metric utilities for use by the ops implementation
pub use metric_p_value as distance_metric_p_value;
pub use metric_to_index as distance_metric_to_index;
