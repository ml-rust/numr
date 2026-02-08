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
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};
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
    let lu_guard = AllocGuard::new(client.allocator(), lu_size)?;
    let lu_ptr = lu_guard.ptr();
    let lu_buffer =
        get_buffer(lu_ptr).ok_or_else(|| Error::Internal("Failed to get LU buffer".to_string()))?;

    let pivots_size = k * std::mem::size_of::<i32>();
    let pivots_guard = AllocGuard::new(client.allocator(), pivots_size)?;
    let pivots_ptr = pivots_guard.ptr();
    let pivots_buffer = get_buffer(pivots_ptr)
        .ok_or_else(|| Error::Internal("Failed to get pivots buffer".to_string()))?;

    let num_swaps_size = std::mem::size_of::<i32>();
    let num_swaps_guard = AllocGuard::new(client.allocator(), num_swaps_size)?;
    let num_swaps_ptr = num_swaps_guard.ptr();
    let num_swaps_buffer = get_buffer(num_swaps_ptr)
        .ok_or_else(|| Error::Internal("Failed to get num_swaps buffer".to_string()))?;

    let singular_flag_size = std::mem::size_of::<i32>();
    let singular_flag_guard = AllocGuard::new(client.allocator(), singular_flag_size)?;
    let singular_flag_ptr = singular_flag_guard.ptr();
    let singular_flag_buffer = get_buffer(singular_flag_ptr)
        .ok_or_else(|| Error::Internal("Failed to get singular_flag buffer".to_string()))?;

    // Copy input to LU buffer
    WgpuRuntime::copy_within_device(a.storage().ptr(), lu_ptr, lu_size, device)?;

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
    client.read_buffer(&staging, &mut flags)?;

    let num_swaps = flags[0] as usize;
    let singular = flags[1] != 0;

    // Guards will automatically deallocate flag buffers on drop
    drop(num_swaps_guard);
    drop(singular_flag_guard);

    if singular {
        // Guards will deallocate lu and pivots on drop
        drop(lu_guard);
        drop(pivots_guard);
        return Err(Error::Internal(format!(
            "LU decomposition failed: {}x{} matrix is singular (zero pivot encountered)",
            m, n
        )));
    }

    // Keep pivots as I32 tensor directly (WGSL has no i64 support)
    // Create tensors from GPU memory
    let lu = unsafe { WgpuClient::tensor_from_raw(lu_guard.release(), &[m, n], dtype, device) };
    let pivots =
        unsafe { WgpuClient::tensor_from_raw(pivots_guard.release(), &[k], DType::I32, device) };

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
    let l_guard = AllocGuard::new(client.allocator(), l_size)?;
    let l_ptr = l_guard.ptr();
    let l_buffer =
        get_buffer(l_ptr).ok_or_else(|| Error::Internal("Failed to get L buffer".to_string()))?;

    let not_pd_flag_size = std::mem::size_of::<i32>();
    let not_pd_flag_guard = AllocGuard::new(client.allocator(), not_pd_flag_size)?;
    let not_pd_flag_ptr = not_pd_flag_guard.ptr();
    let not_pd_flag_buffer = get_buffer(not_pd_flag_ptr)
        .ok_or_else(|| Error::Internal("Failed to get not_pd_flag buffer".to_string()))?;

    // Copy input to L buffer
    WgpuRuntime::copy_within_device(a.storage().ptr(), l_ptr, l_size, device)?;

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
    client.read_buffer(&staging, &mut not_pd)?;

    // Guard will automatically deallocate flag buffer on drop
    drop(not_pd_flag_guard);

    if not_pd[0] != 0 {
        // Guard will deallocate l on drop
        drop(l_guard);
        return Err(Error::Internal(format!(
            "Cholesky decomposition failed: {}x{} matrix is not positive definite",
            n, n
        )));
    }

    let l = unsafe { WgpuClient::tensor_from_raw(l_guard.release(), &[n, n], dtype, device) };

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
    let q_guard = AllocGuard::new(client.allocator(), q_size)?;
    let q_ptr = q_guard.ptr();
    let q_buffer =
        get_buffer(q_ptr).ok_or_else(|| Error::Internal("Failed to get Q buffer".to_string()))?;

    // R is [m, n] but only upper triangular part is meaningful
    let r_size = m * n * dtype.size_in_bytes();
    let r_guard = AllocGuard::new(client.allocator(), r_size)?;
    let r_ptr = r_guard.ptr();
    let r_buffer =
        get_buffer(r_ptr).ok_or_else(|| Error::Internal("Failed to get R buffer".to_string()))?;

    // Workspace for Householder vector (size m elements)
    let workspace_size = m * dtype.size_in_bytes();
    let workspace_guard = AllocGuard::new(client.allocator(), workspace_size)?;
    let workspace_ptr = workspace_guard.ptr();
    let workspace_buffer = get_buffer(workspace_ptr)
        .ok_or_else(|| Error::Internal("Failed to get workspace buffer".to_string()))?;

    // Copy A to R (will be modified in place)
    WgpuRuntime::copy_within_device(a.storage().ptr(), r_ptr, r_size, device)?;

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

    // Guard will automatically deallocate workspace on drop
    drop(workspace_guard);

    client.synchronize();

    let q = unsafe { WgpuClient::tensor_from_raw(q_guard.release(), &[m, q_cols], dtype, device) };

    // For thin QR, R should be [k, n]
    let r = if thin && m > n {
        unsafe { WgpuClient::tensor_from_raw(r_guard.release(), &[k, n], dtype, device) }
    } else if thin {
        unsafe { WgpuClient::tensor_from_raw(r_guard.release(), &[m, n], dtype, device) }
    } else {
        unsafe { WgpuClient::tensor_from_raw(r_guard.release(), &[m, n], dtype, device) }
    };

    Ok(QrDecomposition { q, r })
}
