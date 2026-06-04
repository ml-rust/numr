//! WGSL kernel launcher for the GEMM epilogue backward pass. F32 only.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache};
use crate::dtype::DType;
use crate::error::{Error, Result};

const GEMM_EPILOGUE_BWD_SHADER: &str = include_str!("gemm_epilogue_bwd_f32.wgsl");

const TILE_SIZE: u32 = 16;
const BIAS_WG: u32 = 256;

/// Buffers used by the backward pass, in bind-group order.
///
/// All four backward kernels share one bind-group layout (8 storage + 1
/// uniform); each kernel references only the bindings it needs.
#[allow(clippy::too_many_arguments)]
pub struct GemmEpilogueBwdBuffers<'a> {
    /// Forward input A `[batch, M, K]`.
    pub a: &'a Buffer,
    /// Forward weight B `[batch, K, N]`.
    pub b: &'a Buffer,
    /// Forward bias `[N]`.
    pub bias: &'a Buffer,
    /// Upstream gradient `[batch, M, N]`.
    pub grad: &'a Buffer,
    /// Scratch for `grad * activation'(pre)` `[batch, M, N]`.
    pub grad_pre: &'a Buffer,
    /// Output gradient w.r.t. A `[batch, M, K]`.
    pub d_a: &'a Buffer,
    /// Output gradient w.r.t. B `[K, N]` (summed over batch).
    pub d_b: &'a Buffer,
    /// Output gradient w.r.t. bias `[N]` (summed over batch).
    pub d_bias: &'a Buffer,
}

/// Launch the fused GEMM + bias + activation backward pass.
///
/// Records four sequential compute passes in a single encoder:
/// `grad_pre`, then `d_a`, `d_b`, `d_bias`. Pass ordering guarantees the
/// later passes observe the `grad_pre` writes.
#[allow(clippy::too_many_arguments)]
pub fn launch_gemm_bias_activation_bwd(
    cache: &PipelineCache,
    queue: &Queue,
    buffers: &GemmEpilogueBwdBuffers,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    k: usize,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "gemm_bias_activation_bwd",
        });
    }

    let module = cache.get_or_create_module("gemm_epilogue_bwd_f32", GEMM_EPILOGUE_BWD_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 8,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            buffers.a,
            buffers.b,
            buffers.bias,
            buffers.grad,
            buffers.grad_pre,
            buffers.d_a,
            buffers.d_b,
            buffers.d_bias,
            params_buffer,
        ],
    );

    let grad_pre_pipeline = cache.get_or_create_pipeline(
        "gemm_epilogue_bwd_f32",
        "gemm_bwd_grad_pre_f32",
        &module,
        &layout,
    );
    let da_pipeline =
        cache.get_or_create_pipeline("gemm_epilogue_bwd_f32", "gemm_bwd_da_f32", &module, &layout);
    let db_pipeline =
        cache.get_or_create_pipeline("gemm_epilogue_bwd_f32", "gemm_bwd_db_f32", &module, &layout);
    let dbias_pipeline = cache.get_or_create_pipeline(
        "gemm_epilogue_bwd_f32",
        "gemm_bwd_dbias_f32",
        &module,
        &layout,
    );

    let (m_u, n_u, k_u, batch_u) = (m as u32, n as u32, k as u32, batch_size as u32);
    let tiles = |x: u32| x.div_ceil(TILE_SIZE);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gemm_bias_activation_bwd"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm_bwd_grad_pre"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&grad_pre_pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(tiles(n_u), tiles(m_u), batch_u);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm_bwd_da"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&da_pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(tiles(k_u), tiles(m_u), batch_u);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm_bwd_db"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&db_pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(tiles(n_u), tiles(k_u), 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm_bwd_dbias"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&dbias_pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(n_u.div_ceil(BIAS_WG), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
