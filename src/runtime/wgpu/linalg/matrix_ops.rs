//! Matrix operations: inverse, determinant, trace, diagonal, rank, and norm.

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use crate::algorithm::linalg::{
    MatrixNormOrder, validate_linalg_dtype, validate_matrix_2d, validate_square_matrix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::TensorOps;
use crate::runtime::{Allocator, Runtime, RuntimeClient};
use crate::tensor::Tensor;

pub fn inverse(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU inverse (only F32 supported)",
        });
    }

    // Compute LU decomposition using decompositions module
    use super::decompositions::lu_decompose;
    let lu_result = lu_decompose(client, a)?;

    // Allocate output and temporary buffers
    let inv_size = n * n * dtype.size_in_bytes();
    let inv_ptr = client.allocator().allocate(inv_size);
    let inv_buffer = get_buffer(inv_ptr)
        .ok_or_else(|| Error::Internal("Failed to get inv buffer".to_string()))?;

    let col_size = n * dtype.size_in_bytes();

    // Create identity matrix on GPU
    let identity_ptr = client.allocator().allocate(inv_size);
    let identity_buffer = get_buffer(identity_ptr)
        .ok_or_else(|| Error::Internal("Failed to get identity buffer".to_string()))?;

    let id_params: [u32; 1] = [n as u32];
    let id_params_buffer = client.create_uniform_buffer("identity_params", 4);
    client.write_buffer(&id_params_buffer, &id_params);

    kernels::launch_create_identity(
        client.pipeline_cache(),
        &client.queue,
        &identity_buffer,
        &id_params_buffer,
        n,
        dtype,
    )?;

    // Get LU and pivots buffers
    let lu_buffer = get_buffer(lu_result.lu.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get lu buffer".to_string()))?;

    // Convert pivots to i32
    let pivots_i64: Vec<i64> = lu_result.pivots.to_vec();
    let pivots_i32: Vec<i32> = pivots_i64.iter().map(|&p| p as i32).collect();
    let pivots_ptr = client.allocator().allocate(n * std::mem::size_of::<i32>());
    WgpuRuntime::copy_to_device(bytemuck::cast_slice(&pivots_i32), pivots_ptr, device);
    let pivots_buffer = get_buffer(pivots_ptr)
        .ok_or_else(|| Error::Internal("Failed to get pivots buffer".to_string()))?;

    // Allocate temporary buffers
    let e_ptr = client.allocator().allocate(col_size);
    let pb_ptr = client.allocator().allocate(col_size);
    let y_ptr = client.allocator().allocate(col_size);
    let x_ptr = client.allocator().allocate(col_size);

    let e_buffer =
        get_buffer(e_ptr).ok_or_else(|| Error::Internal("Failed to get e buffer".to_string()))?;
    let pb_buffer =
        get_buffer(pb_ptr).ok_or_else(|| Error::Internal("Failed to get pb buffer".to_string()))?;
    let y_buffer =
        get_buffer(y_ptr).ok_or_else(|| Error::Internal("Failed to get y buffer".to_string()))?;
    let x_buffer =
        get_buffer(x_ptr).ok_or_else(|| Error::Internal("Failed to get x buffer".to_string()))?;

    // Solve for each column of the identity matrix
    for col in 0..n {
        // Extract column from identity matrix
        let extract_params: [u32; 3] = [n as u32, n as u32, col as u32];
        let extract_params_buffer = client.create_uniform_buffer("extract_params", 12);
        client.write_buffer(&extract_params_buffer, &extract_params);

        kernels::launch_extract_column(
            client.pipeline_cache(),
            &client.queue,
            &identity_buffer,
            &e_buffer,
            &extract_params_buffer,
            n,
            dtype,
        )?;

        // Apply permutation: pb = P @ e
        let perm_params: [u32; 1] = [n as u32];
        let perm_params_buffer = client.create_uniform_buffer("perm_params", 4);
        client.write_buffer(&perm_params_buffer, &perm_params);

        kernels::launch_apply_lu_permutation(
            client.pipeline_cache(),
            &client.queue,
            &e_buffer,
            &pb_buffer,
            &pivots_buffer,
            &perm_params_buffer,
            dtype,
        )?;

        // Forward substitution: Ly = pb (L has unit diagonal)
        let forward_params: [u32; 2] = [n as u32, 1];
        let forward_params_buffer = client.create_uniform_buffer("forward_params", 8);
        client.write_buffer(&forward_params_buffer, &forward_params);

        kernels::launch_forward_sub(
            client.pipeline_cache(),
            &client.queue,
            &lu_buffer,
            &pb_buffer,
            &y_buffer,
            &forward_params_buffer,
            dtype,
        )?;

        // Backward substitution: Ux = y
        let backward_params: [u32; 1] = [n as u32];
        let backward_params_buffer = client.create_uniform_buffer("backward_params", 4);
        client.write_buffer(&backward_params_buffer, &backward_params);

        kernels::launch_backward_sub(
            client.pipeline_cache(),
            &client.queue,
            &lu_buffer,
            &y_buffer,
            &x_buffer,
            &backward_params_buffer,
            dtype,
        )?;

        // Scatter x into column of inverse matrix
        let scatter_params: [u32; 2] = [n as u32, col as u32];
        let scatter_params_buffer = client.create_uniform_buffer("scatter_params", 8);
        client.write_buffer(&scatter_params_buffer, &scatter_params);

        kernels::launch_scatter_column(
            client.pipeline_cache(),
            &client.queue,
            &x_buffer,
            &inv_buffer,
            &scatter_params_buffer,
            n,
            dtype,
        )?;
    }

    client.synchronize();

    // Clean up
    client.allocator().deallocate(identity_ptr, inv_size);
    client
        .allocator()
        .deallocate(pivots_ptr, n * std::mem::size_of::<i32>());
    client.allocator().deallocate(e_ptr, col_size);
    client.allocator().deallocate(pb_ptr, col_size);
    client.allocator().deallocate(y_ptr, col_size);
    client.allocator().deallocate(x_ptr, col_size);

    let inv = unsafe { WgpuClient::tensor_from_raw(inv_ptr, &[n, n], dtype, device) };

    Ok(inv)
}

pub fn det(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU det (only F32 supported)",
        });
    }

    // Compute LU decomposition
    use super::decompositions::lu_decompose;
    let lu_result = lu_decompose(client, a)?;

    // Allocate output
    let det_size = dtype.size_in_bytes();
    let det_ptr = client.allocator().allocate(det_size);
    let det_buffer = get_buffer(det_ptr)
        .ok_or_else(|| Error::Internal("Failed to get det buffer".to_string()))?;

    let lu_buffer = get_buffer(lu_result.lu.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get lu buffer".to_string()))?;

    // Create params buffer with n and num_swaps
    let params: [u32; 2] = [n as u32, lu_result.num_swaps as u32];
    let params_buffer = client.create_uniform_buffer("det_params", 8);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_det_from_lu(
        client.pipeline_cache(),
        &client.queue,
        &lu_buffer,
        &det_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    let det = unsafe { WgpuClient::tensor_from_raw(det_ptr, &[], dtype, device) };

    Ok(det)
}

pub fn trace(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let min_dim = m.min(n);
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU trace (only F32 supported)",
        });
    }

    // Allocate output (zero-initialized for reduction)
    let trace_size = dtype.size_in_bytes();
    let trace_ptr = client.allocator().allocate(trace_size);
    let trace_buffer = get_buffer(trace_ptr)
        .ok_or_else(|| Error::Internal("Failed to get trace buffer".to_string()))?;

    let zero: [f32; 1] = [0.0];
    client.write_buffer(&trace_buffer, &zero);

    let a_buffer = get_buffer(a.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get a buffer".to_string()))?;

    let params: [u32; 2] = [min_dim as u32, n as u32];
    let params_buffer = client.create_uniform_buffer("trace_params", 8);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_trace(
        client.pipeline_cache(),
        &client.queue,
        &a_buffer,
        &trace_buffer,
        &params_buffer,
        min_dim,
        dtype,
    )?;

    client.synchronize();

    let trace = unsafe { WgpuClient::tensor_from_raw(trace_ptr, &[], dtype, device) };

    Ok(trace)
}

pub fn diag(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let min_dim = m.min(n);
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU diag (only F32 supported)",
        });
    }

    let diag_size = min_dim * dtype.size_in_bytes();
    let diag_ptr = client.allocator().allocate(diag_size);
    let diag_buffer = get_buffer(diag_ptr)
        .ok_or_else(|| Error::Internal("Failed to get diag buffer".to_string()))?;

    let a_buffer = get_buffer(a.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get a buffer".to_string()))?;

    let params: [u32; 2] = [min_dim as u32, n as u32];
    let params_buffer = client.create_uniform_buffer("diag_params", 8);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_diag(
        client.pipeline_cache(),
        &client.queue,
        &a_buffer,
        &diag_buffer,
        &params_buffer,
        min_dim,
        dtype,
    )?;

    client.synchronize();

    let diag = unsafe { WgpuClient::tensor_from_raw(diag_ptr, &[min_dim], dtype, device) };

    Ok(diag)
}

pub fn diagflat(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;

    if a.shape().len() != 1 {
        return Err(Error::Internal(format!(
            "diagflat requires 1D input tensor, got {}D tensor with shape {:?}",
            a.shape().len(),
            a.shape()
        )));
    }

    let n = a.shape()[0];
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU diagflat (only F32 supported)",
        });
    }

    let out_size = n * n * dtype.size_in_bytes();
    let out_ptr = client.allocator().allocate(out_size);
    let out_buffer = get_buffer(out_ptr)
        .ok_or_else(|| Error::Internal("Failed to get out buffer".to_string()))?;

    let a_buffer = get_buffer(a.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get a buffer".to_string()))?;

    let params: [u32; 1] = [n as u32];
    let params_buffer = client.create_uniform_buffer("diagflat_params", 4);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_diagflat(
        client.pipeline_cache(),
        &client.queue,
        &a_buffer,
        &out_buffer,
        &params_buffer,
        n,
        dtype,
    )?;

    client.synchronize();

    let out = unsafe { WgpuClient::tensor_from_raw(out_ptr, &[n, n], dtype, device) };

    Ok(out)
}

pub fn matrix_rank(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    tol: Option<f64>,
) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();
    let k = m.min(n);

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU matrix_rank (only F32 supported)",
        });
    }

    // Use QR decomposition to estimate rank
    use super::decompositions::qr_decompose_internal;
    let qr = qr_decompose_internal(client, a, false)?;

    // Get diagonal of R
    let r_diag = TensorOps::diag(client, &qr.r)?;

    // Allocate GPU buffers for max abs and count
    let max_size = dtype.size_in_bytes();
    let max_ptr = client.allocator().allocate(max_size);
    let max_buffer = get_buffer(max_ptr)
        .ok_or_else(|| Error::Internal("Failed to get max buffer".to_string()))?;

    let count_size = std::mem::size_of::<u32>();
    let count_ptr = client.allocator().allocate(count_size);
    let count_buffer = get_buffer(count_ptr)
        .ok_or_else(|| Error::Internal("Failed to get count buffer".to_string()))?;

    let r_diag_buffer = get_buffer(r_diag.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get r_diag buffer".to_string()))?;

    // Zero-initialize
    let zero_f32: [f32; 1] = [0.0];
    let zero_u32: [u32; 1] = [0];
    client.write_buffer(&max_buffer, &zero_f32);
    client.write_buffer(&count_buffer, &zero_u32);

    // Compute max absolute value
    let max_params: [u32; 1] = [k as u32];
    let max_params_buffer = client.create_uniform_buffer("max_params", 4);
    client.write_buffer(&max_params_buffer, &max_params);

    kernels::launch_max_abs(
        client.pipeline_cache(),
        &client.queue,
        &r_diag_buffer,
        &max_buffer,
        &max_params_buffer,
        k,
        dtype,
    )?;

    client.synchronize();

    // Read max value
    let staging = client.create_staging_buffer("max_staging", 4);
    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("max_copy"),
        });
    encoder.copy_buffer_to_buffer(&max_buffer, 0, &staging, 0, 4);
    client.submit_and_wait(encoder);

    let mut max_data = [0.0f32; 1];
    client.read_buffer(&staging, &mut max_data);
    let max_diag = max_data[0] as f64;

    // Compute tolerance
    let base_tol = tol.unwrap_or_else(|| {
        let eps = f32::EPSILON as f64;
        (m.max(n) as f64) * eps
    });
    let threshold = (base_tol * max_diag) as f32;

    // Count elements above threshold
    let count_params_buffer = client.create_uniform_buffer("count_params", 8);
    // Pack k and threshold into uniform buffer
    client
        .queue
        .write_buffer(&count_params_buffer, 0, bytemuck::cast_slice(&[k as u32]));
    client
        .queue
        .write_buffer(&count_params_buffer, 4, bytemuck::cast_slice(&[threshold]));

    kernels::launch_count_above_threshold(
        client.pipeline_cache(),
        &client.queue,
        &r_diag_buffer,
        &count_buffer,
        &count_params_buffer,
        k,
        dtype,
    )?;

    client.synchronize();

    // Read count
    let staging_count = client.create_staging_buffer("count_staging", 4);
    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("count_copy"),
        });
    encoder.copy_buffer_to_buffer(&count_buffer, 0, &staging_count, 0, 4);
    client.submit_and_wait(encoder);

    let mut count_data = [0u32; 1];
    client.read_buffer(&staging_count, &mut count_data);
    let rank = count_data[0] as i64;

    // Clean up
    client.allocator().deallocate(max_ptr, max_size);
    client.allocator().deallocate(count_ptr, count_size);

    // Create rank tensor
    let rank_data: [i64; 1] = [rank];
    let rank_tensor = Tensor::<WgpuRuntime>::from_slice(&rank_data, &[], device);

    Ok(rank_tensor)
}

pub fn matrix_norm(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    ord: MatrixNormOrder,
) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (_m, _n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU matrix_norm (only F32 supported)",
        });
    }

    match ord {
        MatrixNormOrder::Frobenius => {
            // Frobenius norm: ||A||_F = sqrt(sum(A²))
            // Use existing tensor ops to keep data on GPU
            let squared = client.square(a)?;
            let sum_sq = client.sum(&squared, &[], false)?;
            client.sqrt(&sum_sq)
        }
        MatrixNormOrder::Spectral => {
            // Spectral norm is the largest singular value
            use super::svd::svd_decompose;
            let svd = svd_decompose(client, a)?;
            // S is already sorted descending, so s[0] is the largest
            let s_vec: Vec<f32> = svd.s.to_vec();
            let spectral_val = s_vec.first().copied().unwrap_or(0.0);
            Ok(Tensor::<WgpuRuntime>::from_slice(
                &[spectral_val],
                &[],
                a.device(),
            ))
        }
        MatrixNormOrder::Nuclear => {
            // Nuclear norm is sum of singular values
            use super::svd::svd_decompose;
            let svd = svd_decompose(client, a)?;
            client.sum(&svd.s, &[], false)
        }
    }
}

/// Kronecker product: A ⊗ B
pub fn kron(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

    let (m_a, n_a) = validate_matrix_2d(a.shape())?;
    let (m_b, n_b) = validate_matrix_2d(b.shape())?;

    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU kron (only F32 supported)",
        });
    }

    let m_out = m_a * m_b;
    let n_out = n_a * n_b;
    let out_size = m_out * n_out * dtype.size_in_bytes();
    let out_ptr = client.allocator().allocate(out_size);
    let out_buffer = get_buffer(out_ptr)
        .ok_or_else(|| Error::Internal("Failed to get out buffer".to_string()))?;

    let a_buffer = get_buffer(a.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get a buffer".to_string()))?;

    let b_buffer = get_buffer(b.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get b buffer".to_string()))?;

    let params: [u32; 4] = [m_a as u32, n_a as u32, m_b as u32, n_b as u32];
    let params_buffer = client.create_uniform_buffer("kron_params", 16);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_kron(
        client.pipeline_cache(),
        &client.queue,
        &a_buffer,
        &b_buffer,
        &out_buffer,
        &params_buffer,
        m_a * m_b * n_a * n_b,
        dtype,
    )?;

    client.synchronize();

    let out = unsafe { WgpuClient::tensor_from_raw(out_ptr, &[m_out, n_out], dtype, device) };

    Ok(out)
}
