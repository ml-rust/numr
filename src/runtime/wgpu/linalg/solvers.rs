//! Triangular and linear system solvers for WebGPU backend.

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use super::decompositions::qr_decompose_internal;
use crate::algorithm::linalg::{validate_linalg_dtype, validate_matrix_2d, validate_square_matrix};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{LinalgOps, MatmulOps};
use crate::runtime::{Allocator, Runtime, RuntimeClient};
use crate::tensor::Tensor;

pub fn solve(
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
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU solve (only F32 supported)",
        });
    }

    // Only handle vector b for now
    let b_shape = b.shape();
    if b_shape.len() != 1 || b_shape[0] != n {
        return Err(Error::ShapeMismatch {
            expected: vec![n],
            got: b_shape.to_vec(),
        });
    }

    // Compute LU decomposition using decompositions module
    use super::decompositions::lu_decompose;
    let lu_result = lu_decompose(client, a)?;

    // Allocate output and temporary buffers
    let x_size = n * dtype.size_in_bytes();
    let x_ptr = client.allocator().allocate(x_size);
    let x_buffer =
        get_buffer(x_ptr).ok_or_else(|| Error::Internal("Failed to get x buffer".to_string()))?;

    let pb_ptr = client.allocator().allocate(x_size);
    let pb_buffer =
        get_buffer(pb_ptr).ok_or_else(|| Error::Internal("Failed to get pb buffer".to_string()))?;

    let y_ptr = client.allocator().allocate(x_size);
    let y_buffer =
        get_buffer(y_ptr).ok_or_else(|| Error::Internal("Failed to get y buffer".to_string()))?;

    // Get input buffers
    let b_buffer = get_buffer(b.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get b buffer".to_string()))?;
    let lu_buffer = get_buffer(lu_result.lu.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get lu buffer".to_string()))?;

    // Convert pivots to i32 for shader
    let pivots_i64: Vec<i64> = lu_result.pivots.to_vec();
    let pivots_i32: Vec<i32> = pivots_i64.iter().map(|&p| p as i32).collect();
    let pivots_ptr = client.allocator().allocate(n * std::mem::size_of::<i32>());
    WgpuRuntime::copy_to_device(bytemuck::cast_slice(&pivots_i32), pivots_ptr, device);
    let pivots_buffer = get_buffer(pivots_ptr)
        .ok_or_else(|| Error::Internal("Failed to get pivots buffer".to_string()))?;

    // Apply permutation: pb = P @ b
    let perm_params: [u32; 1] = [n as u32];
    let perm_params_buffer = client.create_uniform_buffer("perm_params", 4);
    client.write_buffer(&perm_params_buffer, &perm_params);

    kernels::launch_apply_lu_permutation(
        client.pipeline_cache(),
        &client.queue,
        &b_buffer,
        &pb_buffer,
        &pivots_buffer,
        &perm_params_buffer,
        dtype,
    )?;

    // Forward substitution: Ly = pb (L has unit diagonal)
    let forward_params: [u32; 2] = [n as u32, 1]; // unit_diagonal = 1
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

    client.synchronize();

    // Clean up temporary buffers
    client.allocator().deallocate(pb_ptr, x_size);
    client.allocator().deallocate(y_ptr, x_size);
    client
        .allocator()
        .deallocate(pivots_ptr, n * std::mem::size_of::<i32>());

    let x = unsafe { WgpuClient::tensor_from_raw(x_ptr, &[n], dtype, device) };

    Ok(x)
}

pub fn solve_triangular_lower(
    client: &WgpuClient,
    l: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    unit_diagonal: bool,
) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(l.dtype())?;
    if l.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: l.dtype(),
            rhs: b.dtype(),
        });
    }
    let n = validate_square_matrix(l.shape())?;
    let dtype = l.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU solve_triangular_lower (only F32 supported)",
        });
    }

    if b.shape().len() != 1 || b.shape()[0] != n {
        return Err(Error::ShapeMismatch {
            expected: vec![n],
            got: b.shape().to_vec(),
        });
    }

    let x_size = n * dtype.size_in_bytes();
    let x_ptr = client.allocator().allocate(x_size);
    let x_buffer =
        get_buffer(x_ptr).ok_or_else(|| Error::Internal("Failed to get x buffer".to_string()))?;

    let l_buffer = get_buffer(l.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get L buffer".to_string()))?;
    let b_buffer = get_buffer(b.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get b buffer".to_string()))?;

    let params: [u32; 2] = [n as u32, if unit_diagonal { 1 } else { 0 }];
    let params_buffer = client.create_uniform_buffer("forward_params", 8);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_forward_sub(
        client.pipeline_cache(),
        &client.queue,
        &l_buffer,
        &b_buffer,
        &x_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    let x = unsafe { WgpuClient::tensor_from_raw(x_ptr, &[n], dtype, device) };

    Ok(x)
}

pub fn solve_triangular_upper(
    client: &WgpuClient,
    u: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(u.dtype())?;
    if u.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: u.dtype(),
            rhs: b.dtype(),
        });
    }
    let n = validate_square_matrix(u.shape())?;
    let dtype = u.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU solve_triangular_upper (only F32 supported)",
        });
    }

    if b.shape().len() != 1 || b.shape()[0] != n {
        return Err(Error::ShapeMismatch {
            expected: vec![n],
            got: b.shape().to_vec(),
        });
    }

    let x_size = n * dtype.size_in_bytes();
    let x_ptr = client.allocator().allocate(x_size);
    let x_buffer =
        get_buffer(x_ptr).ok_or_else(|| Error::Internal("Failed to get x buffer".to_string()))?;

    let u_buffer = get_buffer(u.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get U buffer".to_string()))?;
    let b_buffer = get_buffer(b.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get b buffer".to_string()))?;

    let params: [u32; 1] = [n as u32];
    let params_buffer = client.create_uniform_buffer("backward_params", 4);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_backward_sub(
        client.pipeline_cache(),
        &client.queue,
        &u_buffer,
        &b_buffer,
        &x_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    let x = unsafe { WgpuClient::tensor_from_raw(x_ptr, &[n], dtype, device) };

    Ok(x)
}

pub fn lstsq(
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

    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU lstsq (only F32 supported)",
        });
    }

    // Only handle vector b for now
    let b_shape = b.shape();
    if b_shape.len() != 1 {
        return Err(Error::Internal(
            "lstsq currently only supports 1D b vector for WGPU".to_string(),
        ));
    }

    // Underdetermined systems not supported
    if m < n {
        return Err(Error::Internal(format!(
            "lstsq: underdetermined system not supported (A is {}x{}, requires m >= n)",
            m, n
        )));
    }

    // QR decomposition
    let qr = qr_decompose_internal(client, a, false)?;

    // Make Q^T contiguous before matmul (transpose creates a view)
    let q_t = qr.q.transpose(0, 1)?.contiguous();
    let b_mat = b.reshape(&[m, 1])?.contiguous();

    // Q^T @ B gives [m, 1]
    let qtb = client.matmul(&q_t, &b_mat)?;

    // Allocate output X [n]
    let x_size = n * dtype.size_in_bytes();
    let x_ptr = client.allocator().allocate(x_size);
    let x_buffer =
        get_buffer(x_ptr).ok_or_else(|| Error::Internal("Failed to get x buffer".to_string()))?;

    // Zero initialize X
    let zero_bytes = vec![0u8; x_size];
    WgpuRuntime::copy_to_device(&zero_bytes, x_ptr, device);

    // Get first n elements of Q^T @ b
    // Make contiguous before reading since matmul output may not be contiguous
    let qtb_contig = qtb.contiguous();
    let qtb_data: Vec<f32> = qtb_contig.to_vec();
    let qtb_n: Vec<f32> = qtb_data[..n].to_vec();
    let qtb_ptr = client.allocator().allocate(x_size);
    WgpuRuntime::copy_to_device(bytemuck::cast_slice(&qtb_n), qtb_ptr, device);
    let qtb_buffer = get_buffer(qtb_ptr)
        .ok_or_else(|| Error::Internal("Failed to get qtb buffer".to_string()))?;

    let r_buffer = get_buffer(qr.r.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get R buffer".to_string()))?;

    // Backward substitution: R @ x = qtb[:n]
    let params: [u32; 1] = [n as u32];
    let params_buffer = client.create_uniform_buffer("backward_params", 4);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_backward_sub(
        client.pipeline_cache(),
        &client.queue,
        &r_buffer,
        &qtb_buffer,
        &x_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    client.allocator().deallocate(qtb_ptr, x_size);

    let x = unsafe { WgpuClient::tensor_from_raw(x_ptr, &[n], dtype, device) };

    Ok(x)
}
