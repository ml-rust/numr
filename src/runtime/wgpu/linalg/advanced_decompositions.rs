//! Advanced decomposition algorithms for WebGPU: rsf2csf, QZ, and polar decomposition
//!
//! All algorithms use native WGSL compute shaders - NO CPU FALLBACK.

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use super::helpers::get_buffer_or_err;
use crate::algorithm::linalg::{
    ComplexSchurDecomposition, GeneralizedSchurDecomposition, LinearAlgebraAlgorithms,
    PolarDecomposition, SchurDecomposition, validate_linalg_dtype, validate_square_matrix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::TensorOps;
use crate::runtime::{Allocator, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Convert real Schur form to complex Schur form (rsf2csf)
///
/// Uses native WGSL shader to process 2x2 blocks representing complex
/// conjugate eigenvalue pairs and converts them to upper triangular form.
pub fn rsf2csf(
    client: &WgpuClient,
    schur: &SchurDecomposition<WgpuRuntime>,
) -> Result<ComplexSchurDecomposition<WgpuRuntime>> {
    let n = validate_square_matrix(schur.t.shape())?;
    let dtype = schur.t.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "rsf2csf (WebGPU)",
        });
    }

    // Handle trivial cases
    if n == 0 {
        return Ok(ComplexSchurDecomposition {
            z_real: Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0, 0], device),
            z_imag: Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0, 0], device),
            t_real: Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0, 0], device),
            t_imag: Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0, 0], device),
        });
    }

    if n == 1 {
        let t_data: Vec<f32> = schur.t.to_vec();
        let z_data: Vec<f32> = schur.z.to_vec();
        return Ok(ComplexSchurDecomposition {
            z_real: Tensor::<WgpuRuntime>::from_slice(&z_data, &[1, 1], device),
            z_imag: Tensor::<WgpuRuntime>::from_slice(&[0.0f32], &[1, 1], device),
            t_real: Tensor::<WgpuRuntime>::from_slice(&t_data, &[1, 1], device),
            t_imag: Tensor::<WgpuRuntime>::from_slice(&[0.0f32], &[1, 1], device),
        });
    }

    let matrix_size = n * n * dtype.size_in_bytes();

    // Allocate buffers for T (real and imag)
    let t_real_ptr = client.allocator().allocate(matrix_size);
    let t_real_buffer = get_buffer_or_err!(t_real_ptr, "T_real");

    let t_imag_ptr = client.allocator().allocate(matrix_size);
    let t_imag_buffer = get_buffer_or_err!(t_imag_ptr, "T_imag");

    // Allocate buffers for Z (real and imag)
    let z_real_ptr = client.allocator().allocate(matrix_size);
    let z_real_buffer = get_buffer_or_err!(z_real_ptr, "Z_real");

    let z_imag_ptr = client.allocator().allocate(matrix_size);
    let z_imag_buffer = get_buffer_or_err!(z_imag_ptr, "Z_imag");

    // Copy input T and Z to real buffers
    WgpuRuntime::copy_within_device(schur.t.storage().ptr(), t_real_ptr, matrix_size, device);
    WgpuRuntime::copy_within_device(schur.z.storage().ptr(), z_real_ptr, matrix_size, device);

    // Zero-initialize imaginary buffers
    let zeros = vec![0.0f32; n * n];
    WgpuRuntime::copy_to_device(bytemuck::cast_slice(&zeros), t_imag_ptr, device);
    WgpuRuntime::copy_to_device(bytemuck::cast_slice(&zeros), z_imag_ptr, device);

    // Create params buffer
    let params: [u32; 1] = [n as u32];
    let params_buffer = client.create_uniform_buffer("rsf2csf_params", 4);
    client.write_buffer(&params_buffer, &params);

    // Launch rsf2csf kernel
    kernels::launch_rsf2csf(
        client.pipeline_cache(),
        &client.queue,
        &t_real_buffer,
        &t_imag_buffer,
        &z_real_buffer,
        &z_imag_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    // Create output tensors
    let z_real = unsafe { WgpuClient::tensor_from_raw(z_real_ptr, &[n, n], dtype, device) };
    let z_imag = unsafe { WgpuClient::tensor_from_raw(z_imag_ptr, &[n, n], dtype, device) };
    let t_real = unsafe { WgpuClient::tensor_from_raw(t_real_ptr, &[n, n], dtype, device) };
    let t_imag = unsafe { WgpuClient::tensor_from_raw(t_imag_ptr, &[n, n], dtype, device) };

    Ok(ComplexSchurDecomposition {
        z_real,
        z_imag,
        t_real,
        t_imag,
    })
}

/// QZ decomposition for matrix pencil (A, B)
///
/// Uses native WGSL shader implementing Hessenberg-triangular reduction
/// followed by QZ iteration.
pub fn qz_decompose(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<GeneralizedSchurDecomposition<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let n_b = validate_square_matrix(b.shape())?;

    if n != n_b {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "qz_decompose (WebGPU)",
        });
    }

    // Handle trivial cases
    if n == 0 {
        return Ok(GeneralizedSchurDecomposition {
            q: Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0, 0], device),
            z: Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0, 0], device),
            s: Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0, 0], device),
            t: Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0, 0], device),
            eigenvalues_real: Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0], device),
            eigenvalues_imag: Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0], device),
        });
    }

    if n == 1 {
        let a_data: Vec<f32> = a.to_vec();
        let b_data: Vec<f32> = b.to_vec();
        let eval = if b_data[0].abs() > 1e-10 {
            a_data[0] / b_data[0]
        } else {
            f32::INFINITY
        };

        return Ok(GeneralizedSchurDecomposition {
            q: Tensor::<WgpuRuntime>::from_slice(&[1.0f32], &[1, 1], device),
            z: Tensor::<WgpuRuntime>::from_slice(&[1.0f32], &[1, 1], device),
            s: Tensor::<WgpuRuntime>::from_slice(&a_data, &[1, 1], device),
            t: Tensor::<WgpuRuntime>::from_slice(&b_data, &[1, 1], device),
            eigenvalues_real: Tensor::<WgpuRuntime>::from_slice(&[eval], &[1], device),
            eigenvalues_imag: Tensor::<WgpuRuntime>::from_slice(&[0.0f32], &[1], device),
        });
    }

    let matrix_size = n * n * dtype.size_in_bytes();
    let vector_size = n * dtype.size_in_bytes();

    // Allocate buffers
    let s_ptr = client.allocator().allocate(matrix_size);
    let s_buffer = get_buffer_or_err!(s_ptr, "S");

    let t_ptr = client.allocator().allocate(matrix_size);
    let t_buffer = get_buffer_or_err!(t_ptr, "T");

    let q_ptr = client.allocator().allocate(matrix_size);
    let q_buffer = get_buffer_or_err!(q_ptr, "Q");

    let z_ptr = client.allocator().allocate(matrix_size);
    let z_buffer = get_buffer_or_err!(z_ptr, "Z");

    let eval_real_ptr = client.allocator().allocate(vector_size);
    let eval_real_buffer = get_buffer_or_err!(eval_real_ptr, "eigenvalues_real");

    let eval_imag_ptr = client.allocator().allocate(vector_size);
    let eval_imag_buffer = get_buffer_or_err!(eval_imag_ptr, "eigenvalues_imag");

    let converged_flag_size = std::mem::size_of::<i32>();
    let converged_flag_ptr = client.allocator().allocate(converged_flag_size);
    let converged_flag_buffer = get_buffer_or_err!(converged_flag_ptr, "QZ convergence flag");

    // Copy input matrices
    WgpuRuntime::copy_within_device(a.storage().ptr(), s_ptr, matrix_size, device);
    WgpuRuntime::copy_within_device(b.storage().ptr(), t_ptr, matrix_size, device);

    // Zero-initialize converged flag
    let zero_i32: [i32; 1] = [0];
    client.write_buffer(&converged_flag_buffer, &zero_i32);

    // Create params buffer
    let params: [u32; 1] = [n as u32];
    let params_buffer = client.create_uniform_buffer("qz_params", 4);
    client.write_buffer(&params_buffer, &params);

    // Launch QZ kernel
    kernels::launch_qz_decompose(
        client.pipeline_cache(),
        &client.queue,
        &s_buffer,
        &t_buffer,
        &q_buffer,
        &z_buffer,
        &eval_real_buffer,
        &eval_imag_buffer,
        &converged_flag_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    // Read back converged flag
    let staging = client.create_staging_buffer("qz_converged_staging", 4);
    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("qz_converged_copy"),
        });
    encoder.copy_buffer_to_buffer(&converged_flag_buffer, 0, &staging, 0, 4);
    client.submit_and_wait(encoder);

    let mut converged_val = [0i32; 1];
    client.read_buffer(&staging, &mut converged_val);

    client
        .allocator()
        .deallocate(converged_flag_ptr, converged_flag_size);

    if converged_val[0] != 0 {
        client.allocator().deallocate(s_ptr, matrix_size);
        client.allocator().deallocate(t_ptr, matrix_size);
        client.allocator().deallocate(q_ptr, matrix_size);
        client.allocator().deallocate(z_ptr, matrix_size);
        client.allocator().deallocate(eval_real_ptr, vector_size);
        client.allocator().deallocate(eval_imag_ptr, vector_size);
        return Err(Error::Internal(
            "QZ decomposition did not converge within maximum iterations".to_string(),
        ));
    }

    // Create output tensors
    let q = unsafe { WgpuClient::tensor_from_raw(q_ptr, &[n, n], dtype, device) };
    let z = unsafe { WgpuClient::tensor_from_raw(z_ptr, &[n, n], dtype, device) };
    let s = unsafe { WgpuClient::tensor_from_raw(s_ptr, &[n, n], dtype, device) };
    let t = unsafe { WgpuClient::tensor_from_raw(t_ptr, &[n, n], dtype, device) };
    let eigenvalues_real =
        unsafe { WgpuClient::tensor_from_raw(eval_real_ptr, &[n], dtype, device) };
    let eigenvalues_imag =
        unsafe { WgpuClient::tensor_from_raw(eval_imag_ptr, &[n], dtype, device) };

    Ok(GeneralizedSchurDecomposition {
        q,
        z,
        s,
        t,
        eigenvalues_real,
        eigenvalues_imag,
    })
}

/// Polar decomposition: A = U @ P
///
/// Uses native SVD decomposition and matrix multiplication.
/// U is unitary/orthogonal, P is symmetric positive semi-definite.
pub fn polar_decompose(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
) -> Result<PolarDecomposition<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "polar_decompose (WebGPU)",
        });
    }

    // Handle trivial cases
    if n == 0 {
        return Ok(PolarDecomposition {
            u: Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0, 0], device),
            p: Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0, 0], device),
        });
    }

    if n == 1 {
        let data: Vec<f32> = a.to_vec();
        let val = data[0];
        let sign = if val >= 0.0 { 1.0f32 } else { -1.0f32 };
        return Ok(PolarDecomposition {
            u: Tensor::<WgpuRuntime>::from_slice(&[sign], &[1, 1], device),
            p: Tensor::<WgpuRuntime>::from_slice(&[val.abs()], &[1, 1], device),
        });
    }

    // Compute SVD: A = U_svd @ S @ V^T
    let svd = client.svd_decompose(a)?;

    // Polar decomposition: A = U_polar @ P
    // where U_polar = U_svd @ V^T (orthogonal/unitary)
    // and P = V @ diag(S) @ V^T (symmetric positive semi-definite)

    // U_polar = U @ V^T
    let u_polar = TensorOps::matmul(client, &svd.u, &svd.vt)?;

    // P = V @ diag(S) @ V^T
    // First, create S as diagonal matrix
    let s_diag = TensorOps::diagflat(client, &svd.s)?;

    // V = Vt^T
    let v = svd.vt.transpose(0, 1)?.contiguous();

    // V @ diag(S)
    let v_s = TensorOps::matmul(client, &v, &s_diag)?;

    // P = (V @ S) @ V^T
    let p = TensorOps::matmul(client, &v_s, &svd.vt)?;

    Ok(PolarDecomposition { u: u_polar, p })
}
