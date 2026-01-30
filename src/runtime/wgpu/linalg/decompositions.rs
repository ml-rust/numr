//! LU, Cholesky, and QR decompositions for WebGPU backend.

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use crate::algorithm::linalg::{
    CholeskyDecomposition, LuDecomposition, QrDecomposition, validate_linalg_dtype,
    validate_matrix_2d, validate_square_matrix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{Allocator, Runtime, RuntimeClient};
use crate::tensor::Tensor;

pub fn lu_decompose(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
) -> Result<LuDecomposition<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let k = m.min(n);
    let dtype = a.dtype();
    let device = client.device();

    // WGSL only supports F32
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU lu_decompose (only F32 supported)",
        });
    }

    // Allocate buffers
    let lu_size = m * n * dtype.size_in_bytes();
    let lu_ptr = client.allocator().allocate(lu_size);
    let lu_buffer =
        get_buffer(lu_ptr).ok_or_else(|| Error::Internal("Failed to get LU buffer".to_string()))?;

    let pivots_size = k * std::mem::size_of::<i32>();
    let pivots_ptr = client.allocator().allocate(pivots_size);
    let pivots_buffer = get_buffer(pivots_ptr)
        .ok_or_else(|| Error::Internal("Failed to get pivots buffer".to_string()))?;

    let num_swaps_size = std::mem::size_of::<i32>();
    let num_swaps_ptr = client.allocator().allocate(num_swaps_size);
    let num_swaps_buffer = get_buffer(num_swaps_ptr)
        .ok_or_else(|| Error::Internal("Failed to get num_swaps buffer".to_string()))?;

    let singular_flag_size = std::mem::size_of::<i32>();
    let singular_flag_ptr = client.allocator().allocate(singular_flag_size);
    let singular_flag_buffer = get_buffer(singular_flag_ptr)
        .ok_or_else(|| Error::Internal("Failed to get singular_flag buffer".to_string()))?;

    // Copy input to LU buffer
    WgpuRuntime::copy_within_device(a.storage().ptr(), lu_ptr, lu_size, device);

    // Create params buffer
    let params: [u32; 2] = [m as u32, n as u32];
    let params_buffer = client.create_uniform_buffer("lu_params", 8);
    client.write_buffer(&params_buffer, &params);

    // Zero-initialize flags
    let zero_i32: [i32; 1] = [0];
    client.write_buffer(&num_swaps_buffer, &zero_i32);
    client.write_buffer(&singular_flag_buffer, &zero_i32);

    // Launch kernel
    kernels::launch_lu_decompose(
        client.pipeline_cache(),
        &client.queue,
        &lu_buffer,
        &pivots_buffer,
        &num_swaps_buffer,
        &singular_flag_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    // Read back flags
    let staging = client.create_staging_buffer("lu_flags_staging", 8);
    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lu_flags_copy"),
        });
    encoder.copy_buffer_to_buffer(&num_swaps_buffer, 0, &staging, 0, 4);
    encoder.copy_buffer_to_buffer(&singular_flag_buffer, 0, &staging, 4, 4);
    client.submit_and_wait(encoder);

    let mut flags = [0i32; 2];
    client.read_buffer(&staging, &mut flags);

    let num_swaps = flags[0] as usize;
    let singular = flags[1] != 0;

    // Clean up flag allocations
    client.allocator().deallocate(num_swaps_ptr, num_swaps_size);
    client
        .allocator()
        .deallocate(singular_flag_ptr, singular_flag_size);

    if singular {
        client.allocator().deallocate(lu_ptr, lu_size);
        client.allocator().deallocate(pivots_ptr, pivots_size);
        return Err(Error::Internal(format!(
            "LU decomposition failed: {}x{} matrix is singular (zero pivot encountered)",
            m, n
        )));
    }

    // Convert i32 pivots to i64
    let pivots_i64_size = k * std::mem::size_of::<i64>();
    let pivots_i64_ptr = client.allocator().allocate(pivots_i64_size);

    // Read i32 pivots and convert to i64
    let staging_pivots = client.create_staging_buffer("pivots_staging", pivots_size as u64);
    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("pivots_copy"),
        });
    encoder.copy_buffer_to_buffer(&pivots_buffer, 0, &staging_pivots, 0, pivots_size as u64);
    client.submit_and_wait(encoder);

    let mut pivots_i32 = vec![0i32; k];
    client.read_buffer(&staging_pivots, &mut pivots_i32);

    let pivots_i64: Vec<i64> = pivots_i32.iter().map(|&p| p as i64).collect();
    WgpuRuntime::copy_to_device(bytemuck::cast_slice(&pivots_i64), pivots_i64_ptr, device);

    client.allocator().deallocate(pivots_ptr, pivots_size);

    // Create tensors from GPU memory
    let lu = unsafe { WgpuClient::tensor_from_raw(lu_ptr, &[m, n], dtype, device) };
    let pivots = unsafe { WgpuClient::tensor_from_raw(pivots_i64_ptr, &[k], DType::I64, device) };

    Ok(LuDecomposition {
        lu,
        pivots,
        num_swaps,
    })
}

pub fn cholesky_decompose(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
) -> Result<CholeskyDecomposition<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU cholesky_decompose (only F32 supported)",
        });
    }

    // Allocate output on GPU
    let l_size = n * n * dtype.size_in_bytes();
    let l_ptr = client.allocator().allocate(l_size);
    let l_buffer =
        get_buffer(l_ptr).ok_or_else(|| Error::Internal("Failed to get L buffer".to_string()))?;

    let not_pd_flag_size = std::mem::size_of::<i32>();
    let not_pd_flag_ptr = client.allocator().allocate(not_pd_flag_size);
    let not_pd_flag_buffer = get_buffer(not_pd_flag_ptr)
        .ok_or_else(|| Error::Internal("Failed to get not_pd_flag buffer".to_string()))?;

    // Copy input to L buffer
    WgpuRuntime::copy_within_device(a.storage().ptr(), l_ptr, l_size, device);

    // Create params buffer
    let params: [u32; 1] = [n as u32];
    let params_buffer = client.create_uniform_buffer("chol_params", 4);
    client.write_buffer(&params_buffer, &params);

    // Zero-initialize flag
    let zero_i32: [i32; 1] = [0];
    client.write_buffer(&not_pd_flag_buffer, &zero_i32);

    // Launch kernel
    kernels::launch_cholesky_decompose(
        client.pipeline_cache(),
        &client.queue,
        &l_buffer,
        &not_pd_flag_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    // Read back flag
    let staging = client.create_staging_buffer("chol_flag_staging", 4);
    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("chol_flag_copy"),
        });
    encoder.copy_buffer_to_buffer(&not_pd_flag_buffer, 0, &staging, 0, 4);
    client.submit_and_wait(encoder);

    let mut not_pd = [0i32; 1];
    client.read_buffer(&staging, &mut not_pd);

    client
        .allocator()
        .deallocate(not_pd_flag_ptr, not_pd_flag_size);

    if not_pd[0] != 0 {
        client.allocator().deallocate(l_ptr, l_size);
        return Err(Error::Internal(format!(
            "Cholesky decomposition failed: {}x{} matrix is not positive definite",
            n, n
        )));
    }

    let l = unsafe { WgpuClient::tensor_from_raw(l_ptr, &[n, n], dtype, device) };

    Ok(CholeskyDecomposition { l })
}

pub fn qr_decompose_internal(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    thin: bool,
) -> Result<QrDecomposition<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let k = m.min(n);
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU qr_decompose (only F32 supported)",
        });
    }

    // Q dimensions: [m, m] for full, [m, k] for thin
    let q_cols = if thin { k } else { m };
    let q_size = m * q_cols * dtype.size_in_bytes();
    let q_ptr = client.allocator().allocate(q_size);
    let q_buffer =
        get_buffer(q_ptr).ok_or_else(|| Error::Internal("Failed to get Q buffer".to_string()))?;

    // R is [m, n] but only upper triangular part is meaningful
    let r_size = m * n * dtype.size_in_bytes();
    let r_ptr = client.allocator().allocate(r_size);
    let r_buffer =
        get_buffer(r_ptr).ok_or_else(|| Error::Internal("Failed to get R buffer".to_string()))?;

    // Workspace for Householder vector (size m elements)
    let workspace_size = m * dtype.size_in_bytes();
    let workspace_ptr = client.allocator().allocate(workspace_size);
    let workspace_buffer = get_buffer(workspace_ptr)
        .ok_or_else(|| Error::Internal("Failed to get workspace buffer".to_string()))?;

    // Copy A to R (will be modified in place)
    WgpuRuntime::copy_within_device(a.storage().ptr(), r_ptr, r_size, device);

    // Create params buffer
    let params: [u32; 3] = [m as u32, n as u32, if thin { 1 } else { 0 }];
    let params_buffer = client.create_uniform_buffer("qr_params", 12);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_qr_decompose(
        client.pipeline_cache(),
        &client.queue,
        &q_buffer,
        &r_buffer,
        &workspace_buffer,
        &params_buffer,
        dtype,
    )?;

    // Clean up workspace
    client.allocator().deallocate(workspace_ptr, workspace_size);

    client.synchronize();

    let q = unsafe { WgpuClient::tensor_from_raw(q_ptr, &[m, q_cols], dtype, device) };

    // For thin QR, R should be [k, n]
    let r = if thin && m > n {
        unsafe { WgpuClient::tensor_from_raw(r_ptr, &[k, n], dtype, device) }
    } else if thin {
        unsafe { WgpuClient::tensor_from_raw(r_ptr, &[m, n], dtype, device) }
    } else {
        unsafe { WgpuClient::tensor_from_raw(r_ptr, &[m, n], dtype, device) }
    };

    Ok(QrDecomposition { q, r })
}
