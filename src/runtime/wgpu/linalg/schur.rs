//! Schur decomposition using QR algorithm.

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use super::helpers::get_buffer_or_err;
use crate::algorithm::linalg::{SchurDecomposition, validate_linalg_dtype, validate_square_matrix};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{Allocator, Runtime, RuntimeClient};
use crate::tensor::Tensor;

pub fn schur_decompose(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
) -> Result<SchurDecomposition<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "schur_decompose (WebGPU)",
        });
    }

    // Handle trivial cases
    if n == 0 {
        return Ok(SchurDecomposition {
            z: Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0, 0], device),
            t: Tensor::<WgpuRuntime>::from_slice::<f32>(&[], &[0, 0], device),
        });
    }

    if n == 1 {
        let data: Vec<f32> = a.to_vec();
        return Ok(SchurDecomposition {
            z: Tensor::<WgpuRuntime>::from_slice(&[1.0f32], &[1, 1], device),
            t: Tensor::<WgpuRuntime>::from_slice(&data, &[1, 1], device),
        });
    }

    // Allocate buffers for Schur decomposition
    let matrix_size = n * n * dtype.size_in_bytes();

    let t_ptr = client.allocator().allocate(matrix_size);
    let t_buffer = get_buffer_or_err!(t_ptr, "T (Schur form matrix)");

    let z_ptr = client.allocator().allocate(matrix_size);
    let z_buffer = get_buffer_or_err!(z_ptr, "Z (orthogonal matrix)");

    let converged_flag_size = std::mem::size_of::<i32>();
    let converged_flag_ptr = client.allocator().allocate(converged_flag_size);
    let converged_flag_buffer = get_buffer_or_err!(converged_flag_ptr, "Schur convergence flag");

    // Copy input to T buffer
    WgpuRuntime::copy_within_device(a.storage().ptr(), t_ptr, matrix_size, device);

    // Zero-initialize converged flag
    let zero_i32: [i32; 1] = [0];
    client.write_buffer(&converged_flag_buffer, &zero_i32);

    // Create params buffer
    let params: [u32; 1] = [n as u32];
    let params_buffer = client.create_uniform_buffer("schur_params", 4);
    client.write_buffer(&params_buffer, &params);

    // Launch Schur decomposition kernel
    kernels::launch_schur_decompose(
        client.pipeline_cache(),
        &client.queue,
        &t_buffer,
        &z_buffer,
        &converged_flag_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    // Read back converged flag
    let staging = client.create_staging_buffer("schur_converged_staging", 4);
    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("schur_converged_copy"),
        });
    encoder.copy_buffer_to_buffer(&converged_flag_buffer, 0, &staging, 0, 4);
    client.submit_and_wait(encoder);

    let mut converged_val = [0i32; 1];
    client.read_buffer(&staging, &mut converged_val);

    // Clean up converged flag
    client
        .allocator()
        .deallocate(converged_flag_ptr, converged_flag_size);

    if converged_val[0] != 0 {
        client.allocator().deallocate(t_ptr, matrix_size);
        client.allocator().deallocate(z_ptr, matrix_size);
        return Err(Error::Internal(
            "Schur decomposition did not converge within maximum iterations".to_string(),
        ));
    }

    // Create output tensors from GPU memory
    let z = unsafe { WgpuClient::tensor_from_raw(z_ptr, &[n, n], dtype, device) };
    let t = unsafe { WgpuClient::tensor_from_raw(t_ptr, &[n, n], dtype, device) };

    Ok(SchurDecomposition { z, t })
}
