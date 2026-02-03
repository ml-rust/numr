//! Triangular system solvers for WebGPU backend.
//!
//! # Supported Data Types
//!
//! **F32 only.** WGSL shaders do not natively support F64 operations.

use wgpu::CommandEncoderDescriptor;

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use crate::algorithm::linalg::{validate_linalg_dtype, validate_square_matrix};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{Allocator, RuntimeClient};
use crate::tensor::Tensor;

/// Solve lower triangular system Lx = b
/// Supports both single RHS (b is 1D vector) and multi-RHS (b is 2D matrix [n, nrhs])
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

    // Handle 1D or 2D b
    let b_shape = b.shape();
    let (num_rhs, b_is_vector) = if b_shape.len() == 1 {
        if b_shape[0] != n {
            return Err(Error::ShapeMismatch {
                expected: vec![n],
                got: b_shape.to_vec(),
            });
        }
        (1, true)
    } else if b_shape.len() == 2 {
        if b_shape[0] != n {
            return Err(Error::ShapeMismatch {
                expected: vec![n, b_shape[1]],
                got: b_shape.to_vec(),
            });
        }
        (b_shape[1], false)
    } else {
        return Err(Error::Internal(format!(
            "solve_triangular_lower requires b to be 1D or 2D, got {}D with shape {:?}",
            b_shape.len(),
            b_shape
        )));
    };

    let l_buffer = get_buffer(l.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get L buffer".to_string()))?;
    let b_contig = b.contiguous();
    let b_buffer = get_buffer(b_contig.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get b buffer".to_string()))?;

    // Allocate output
    let x_total_size = n * num_rhs * dtype.size_in_bytes();
    let x_out_ptr = client.allocator().allocate(x_total_size);
    let x_out_buffer = get_buffer(x_out_ptr)
        .ok_or_else(|| Error::Internal("Failed to get x_out buffer".to_string()))?;

    if b_is_vector {
        // Single RHS: direct solve
        let params: [u32; 2] = [n as u32, if unit_diagonal { 1 } else { 0 }];
        let params_buffer = client.create_uniform_buffer("forward_params", 8);
        client.write_buffer(&params_buffer, &params);

        kernels::launch_forward_sub(
            client.pipeline_cache(),
            &client.queue,
            &l_buffer,
            &b_buffer,
            &x_out_buffer,
            &params_buffer,
            dtype,
        )?;
    } else {
        // Multi-RHS: solve for each column
        let col_size = n * dtype.size_in_bytes();
        let col_ptr = client.allocator().allocate(col_size);
        let col_buffer = get_buffer(col_ptr)
            .ok_or_else(|| Error::Internal("Failed to get col buffer".to_string()))?;
        let x_col_ptr = client.allocator().allocate(col_size);
        let x_col_buffer = get_buffer(x_col_ptr)
            .ok_or_else(|| Error::Internal("Failed to get x_col buffer".to_string()))?;

        for rhs in 0..num_rhs {
            // Extract column from B
            let extract_params: [u32; 4] = [n as u32, num_rhs as u32, rhs as u32, 0];
            let extract_params_buffer = client.create_uniform_buffer("extract_params", 16);
            client.write_buffer(&extract_params_buffer, &extract_params);
            kernels::launch_extract_column(
                client.pipeline_cache(),
                &client.queue,
                &b_buffer,
                &col_buffer,
                &extract_params_buffer,
                n,
                dtype,
            )?;

            // Solve L @ x_col = b_col
            let forward_params: [u32; 2] = [n as u32, if unit_diagonal { 1 } else { 0 }];
            let forward_params_buffer = client.create_uniform_buffer("forward_params", 8);
            client.write_buffer(&forward_params_buffer, &forward_params);
            kernels::launch_forward_sub(
                client.pipeline_cache(),
                &client.queue,
                &l_buffer,
                &col_buffer,
                &x_col_buffer,
                &forward_params_buffer,
                dtype,
            )?;

            // Copy x_col to appropriate column in output
            let x_col_offset = rhs * n * dtype.size_in_bytes();
            {
                let mut encoder =
                    client
                        .wgpu_device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("copy_x_col_to_output"),
                        });
                encoder.copy_buffer_to_buffer(
                    &x_col_buffer,
                    0,
                    &x_out_buffer,
                    x_col_offset as u64,
                    col_size as u64,
                );
                client.queue.submit(std::iter::once(encoder.finish()));
            }
        }

        client.allocator().deallocate(col_ptr, col_size);
        client.allocator().deallocate(x_col_ptr, col_size);
    }

    client.synchronize();

    if b_is_vector {
        let x = unsafe { WgpuClient::tensor_from_raw(x_out_ptr, &[n], dtype, device) };
        Ok(x)
    } else {
        // Results stored in column-major order
        let x_col_major =
            unsafe { WgpuClient::tensor_from_raw(x_out_ptr, &[num_rhs, n], dtype, device) };
        let x = x_col_major.transpose(0, 1)?;
        Ok(x.contiguous())
    }
}

/// Solve upper triangular system Ux = b
/// Supports both single RHS (b is 1D vector) and multi-RHS (b is 2D matrix [n, nrhs])
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

    // Handle 1D or 2D b
    let b_shape = b.shape();
    let (num_rhs, b_is_vector) = if b_shape.len() == 1 {
        if b_shape[0] != n {
            return Err(Error::ShapeMismatch {
                expected: vec![n],
                got: b_shape.to_vec(),
            });
        }
        (1, true)
    } else if b_shape.len() == 2 {
        if b_shape[0] != n {
            return Err(Error::ShapeMismatch {
                expected: vec![n, b_shape[1]],
                got: b_shape.to_vec(),
            });
        }
        (b_shape[1], false)
    } else {
        return Err(Error::Internal(format!(
            "solve_triangular_upper requires b to be 1D or 2D, got {}D with shape {:?}",
            b_shape.len(),
            b_shape
        )));
    };

    let u_buffer = get_buffer(u.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get U buffer".to_string()))?;
    let b_contig = b.contiguous();
    let b_buffer = get_buffer(b_contig.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get b buffer".to_string()))?;

    // Allocate output
    let x_total_size = n * num_rhs * dtype.size_in_bytes();
    let x_out_ptr = client.allocator().allocate(x_total_size);
    let x_out_buffer = get_buffer(x_out_ptr)
        .ok_or_else(|| Error::Internal("Failed to get x_out buffer".to_string()))?;

    if b_is_vector {
        // Single RHS: direct solve
        let params: [u32; 1] = [n as u32];
        let params_buffer = client.create_uniform_buffer("backward_params", 4);
        client.write_buffer(&params_buffer, &params);

        kernels::launch_backward_sub(
            client.pipeline_cache(),
            &client.queue,
            &u_buffer,
            &b_buffer,
            &x_out_buffer,
            &params_buffer,
            dtype,
        )?;
    } else {
        // Multi-RHS: solve for each column
        let col_size = n * dtype.size_in_bytes();
        let col_ptr = client.allocator().allocate(col_size);
        let col_buffer = get_buffer(col_ptr)
            .ok_or_else(|| Error::Internal("Failed to get col buffer".to_string()))?;
        let x_col_ptr = client.allocator().allocate(col_size);
        let x_col_buffer = get_buffer(x_col_ptr)
            .ok_or_else(|| Error::Internal("Failed to get x_col buffer".to_string()))?;

        for rhs in 0..num_rhs {
            // Extract column from B
            let extract_params: [u32; 4] = [n as u32, num_rhs as u32, rhs as u32, 0];
            let extract_params_buffer = client.create_uniform_buffer("extract_params", 16);
            client.write_buffer(&extract_params_buffer, &extract_params);
            kernels::launch_extract_column(
                client.pipeline_cache(),
                &client.queue,
                &b_buffer,
                &col_buffer,
                &extract_params_buffer,
                n,
                dtype,
            )?;

            // Solve U @ x_col = b_col
            let backward_params: [u32; 1] = [n as u32];
            let backward_params_buffer = client.create_uniform_buffer("backward_params", 4);
            client.write_buffer(&backward_params_buffer, &backward_params);
            kernels::launch_backward_sub(
                client.pipeline_cache(),
                &client.queue,
                &u_buffer,
                &col_buffer,
                &x_col_buffer,
                &backward_params_buffer,
                dtype,
            )?;

            // Copy x_col to appropriate column in output
            let x_col_offset = rhs * n * dtype.size_in_bytes();
            {
                let mut encoder =
                    client
                        .wgpu_device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("copy_x_col_to_output"),
                        });
                encoder.copy_buffer_to_buffer(
                    &x_col_buffer,
                    0,
                    &x_out_buffer,
                    x_col_offset as u64,
                    col_size as u64,
                );
                client.queue.submit(std::iter::once(encoder.finish()));
            }
        }

        client.allocator().deallocate(col_ptr, col_size);
        client.allocator().deallocate(x_col_ptr, col_size);
    }

    client.synchronize();

    if b_is_vector {
        let x = unsafe { WgpuClient::tensor_from_raw(x_out_ptr, &[n], dtype, device) };
        Ok(x)
    } else {
        // Results stored in column-major order
        let x_col_major =
            unsafe { WgpuClient::tensor_from_raw(x_out_ptr, &[num_rhs, n], dtype, device) };
        let x = x_col_major.transpose(0, 1)?;
        Ok(x.contiguous())
    }
}
