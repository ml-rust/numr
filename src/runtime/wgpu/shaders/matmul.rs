//! Matrix multiplication WGSL kernel launchers
//!
//! Provides launchers for matrix multiplication operations:
//! - 2D matrix multiplication (C = A @ B)
//! - Batched matrix multiplication
//! - Matrix-vector multiplication
//!
//! All operations run entirely on GPU with no CPU fallback.

use wgpu::{Buffer, Queue};

use super::matmul_wgsl::MATMUL_SHADER;
use super::pipeline::{LayoutKey, PipelineCache};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Helper Macros
// ============================================================================

macro_rules! check_dtype_f32 {
    ($dtype:expr, $op:expr) => {
        if $dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype: $dtype,
                op: $op,
            });
        }
    };
}

/// Tile size for tiled matrix multiplication (must match shader constant)
const TILE_SIZE: u32 = 16;

// ============================================================================
// 2D Matrix Multiplication
// ============================================================================

/// Launch tiled matrix multiplication kernel.
///
/// Computes C = A @ B where A is [M, K] and B is [K, N].
pub fn launch_matmul(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "matmul");

    let module = cache.get_or_create_module("matmul", MATMUL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("matmul", "matmul_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // Number of workgroups in x (columns) and y (rows) dimensions
        let num_groups_x = (n as u32 + TILE_SIZE - 1) / TILE_SIZE;
        let num_groups_y = (m as u32 + TILE_SIZE - 1) / TILE_SIZE;
        pass.dispatch_workgroups(num_groups_x, num_groups_y, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch simple (non-tiled) matrix multiplication kernel.
///
/// For small matrices where tiling overhead isn't worth it.
pub fn launch_matmul_simple(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "matmul_simple");

    let module = cache.get_or_create_module("matmul", MATMUL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("matmul", "matmul_simple_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul_simple"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul_simple"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One thread per output element
        let total = m * n;
        let num_groups = (total as u32 + 255) / 256;
        pass.dispatch_workgroups(num_groups, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Batched Matrix Multiplication
// ============================================================================

/// Launch batched matrix multiplication kernel.
///
/// Computes C[b] = A[b] @ B[b] for each batch b.
pub fn launch_batched_matmul(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "batched_matmul");

    let module = cache.get_or_create_module("matmul", MATMUL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("matmul", "batched_matmul_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batched_matmul"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("batched_matmul"),
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

// ============================================================================
// Matrix-Vector Multiplication
// ============================================================================

/// Launch matrix-vector multiplication kernel.
///
/// Computes y = A @ x where A is [M, N] and x is [N].
pub fn launch_matvec(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    x: &Buffer,
    y: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "matvec");

    let module = cache.get_or_create_module("matmul", MATMUL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
    });
    let pipeline = cache.get_or_create_pipeline("matmul", "matvec_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, x, y, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matvec"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matvec"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per output row
        pass.dispatch_workgroups(m as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
