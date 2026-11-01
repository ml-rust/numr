//! Dense × Sparse Matrix Multiplication (DSMM) - WebGPU implementation
//!
//! Implements column-parallel DSMM algorithm for WebGPU backend.
//!
//! # Algorithm
//!
//! Column-parallel CSC iteration (matches CPU/CUDA implementation):
//! ```text
//! For each column j in B (0..N):
//!   For each non-zero B[k,j] in column j:
//!     C[:,j] += A[:,k] * B[k,j]
//! ```
//!
//! WebGPU parallelization:
//! - Each thread computes one element C[row, col]
//! - Thread iterates over sparse column of B to accumulate result
//!
//! This algorithm matches the CPU/CUDA implementation for backend parity.

use super::super::ops::helpers::get_tensor_buffer;
use super::super::shaders::launch_dsmm_csc;
use super::common::validate_wgpu_dtype;
use super::indices::{bindable, indices_to_i32};
use crate::algorithm::sparse::{validate_dsmm_shapes, validate_dtype_match};
use crate::error::Result;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::sparse::CscData;
use crate::tensor::Tensor;

/// DSMM parameters uniform buffer layout.
/// Must match the WGSL struct `DsmmParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DsmmParams {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub _pad: u32,
}

/// Public function to be called from the combined trait implementation
pub(super) fn column_parallel_dsmm(
    client: &WgpuClient,
    dense_a: &Tensor<WgpuRuntime>,
    sparse_b_csc: &CscData<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = dense_a.dtype();
    let device = dense_a.device();

    // Validate dtype
    validate_wgpu_dtype(dtype, "column_parallel_dsmm")?;
    // The check reads dtype metadata on the host. No data moves.
    validate_dtype_match(dtype, sparse_b_csc.values.dtype())?;

    // Validate dimensions
    let ([m, n], k) = validate_dsmm_shapes(dense_a.shape(), sparse_b_csc.shape)?;

    // Ensure A is contiguous for kernel
    let a_contig = dense_a.contiguous()?;

    // Allocate output C [M, N]
    let output = Tensor::<WgpuRuntime>::zeros(&[m, n], dtype, device)?;
    if m == 0 || n == 0 {
        return Ok(output);
    }

    // WGSL reads i32 indices: cast I64 on the GPU.
    let op = "dsmm (WebGPU)";
    let col_ptrs_i32 = indices_to_i32(client, &sparse_b_csc.col_ptrs, op)?;
    let row_indices_i32 = bindable(
        client,
        &indices_to_i32(client, &sparse_b_csc.row_indices, op)?,
    )?;
    let b_values = bindable(client, &sparse_b_csc.values)?;
    let a_contig = bindable(client, &a_contig)?;

    // Create params buffer
    let params = DsmmParams {
        m: m as u32,
        k: k as u32,
        n: n as u32,
        _pad: 0,
    };
    let params_buffer = client.create_uniform_buffer("dsmm_params", 16);
    client.write_buffer(&params_buffer, &[params.m, params.k, params.n, params._pad]);

    // Get buffers
    let a_buffer = get_tensor_buffer(&a_contig)?;
    let col_ptrs_buffer = get_tensor_buffer(&col_ptrs_i32)?;
    let row_indices_buffer = get_tensor_buffer(&row_indices_i32)?;
    let b_values_buffer = get_tensor_buffer(&b_values)?;
    let c_buffer = get_tensor_buffer(&output)?;

    // Launch WGSL kernel
    launch_dsmm_csc(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buffer,
        &col_ptrs_buffer,
        &row_indices_buffer,
        &b_values_buffer,
        &c_buffer,
        &params_buffer,
        m,
        n,
        dtype,
    )?;

    Ok(output)
}
