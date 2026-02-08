//! Matrix multiplication WGSL kernel launchers
//!
//! Provides launchers for matrix multiplication operations:
//! - 2D matrix multiplication (C = A @ B)
//! - Batched matrix multiplication
//! - Matrix-vector multiplication
//! - Fused matmul with bias (C = A @ B + bias)
//!
//! All operations run entirely on GPU with no CPU fallback.

use wgpu::{Buffer, Queue};

use super::generator::generate_matmul_bias_shader;
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
        num_readonly_storage: 0,
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
        num_readonly_storage: 0,
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
        num_readonly_storage: 0,
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
        num_readonly_storage: 0,
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

// ============================================================================
// Fused Matrix Multiplication with Bias
// ============================================================================

/// Helper to get static module key and entry point for matmul_bias
fn matmul_bias_keys(dtype: DType) -> Result<(&'static str, &'static str, &'static str)> {
    match dtype {
        DType::F32 => Ok((
            "matmul_bias_f32",
            "matmul_bias_f32",
            "batched_matmul_bias_f32",
        )),
        DType::I32 => Ok((
            "matmul_bias_i32",
            "matmul_bias_i32",
            "batched_matmul_bias_i32",
        )),
        DType::U32 => Ok((
            "matmul_bias_u32",
            "matmul_bias_u32",
            "batched_matmul_bias_u32",
        )),
        DType::F16 => Ok((
            "matmul_bias_f16",
            "matmul_bias_f16",
            "batched_matmul_bias_f16",
        )),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "matmul_bias",
        }),
    }
}

/// Launch tiled matrix multiplication with fused bias addition.
///
/// Computes C = A @ B + bias where:
/// - A is [M, K]
/// - B is [K, N]
/// - bias is [N] (broadcast across rows)
/// - C is [M, N]
///
/// The bias addition is fused into the GEMM epilogue for efficiency.
pub fn launch_matmul_bias(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    bias: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    dtype: DType,
) -> Result<()> {
    // Get static keys and generate shader
    let (module_key, entry_point, _) = matmul_bias_keys(dtype)?;
    let shader_source = generate_matmul_bias_shader(dtype)?;

    let module = cache.get_or_create_module(module_key, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4, // a, b, bias, c
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    // Bind buffers: a, b, bias, c, params
    let bind_group = cache.create_bind_group(&layout, &[a, b, bias, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul_bias"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul_bias"),
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

/// Launch batched matrix multiplication with fused bias addition.
///
/// Computes C[b] = A[b] @ B[b] + bias for each batch b.
/// The same bias vector is used for all batches.
pub fn launch_batched_matmul_bias(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    bias: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    // Get static keys and generate shader
    let (module_key, _, batched_entry_point) = matmul_bias_keys(dtype)?;
    let shader_source = generate_matmul_bias_shader(dtype)?;

    let module = cache.get_or_create_module(module_key, &shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4, // a, b, bias, c
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, batched_entry_point, &module, &layout);

    // Bind buffers: a, b, bias, c, params
    let bind_group = cache.create_bind_group(&layout, &[a, b, bias, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batched_matmul_bias"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("batched_matmul_bias"),
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
