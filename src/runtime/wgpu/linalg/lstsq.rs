//! Least squares solver for WebGPU backend.
//!
//! # Supported Data Types
//!
//! **F32 only.** WGSL shaders do not natively support F64 operations.

use wgpu::CommandEncoderDescriptor;

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use super::decompositions::qr_decompose_internal;
use crate::algorithm::linalg::{validate_linalg_dtype, validate_matrix_2d};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::MatmulOps;
use crate::runtime::{Allocator, RuntimeClient};
use crate::tensor::Tensor;

/// Least squares solution
/// Supports both single RHS (b is 1D vector) and multi-RHS (b is 2D matrix [m, nrhs])
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

    // Handle 1D or 2D b
    let b_shape = b.shape();
    let (num_rhs, b_is_vector) = if b_shape.len() == 1 {
        if b_shape[0] != m {
            return Err(Error::ShapeMismatch {
                expected: vec![m],
                got: b_shape.to_vec(),
            });
        }
        (1, true)
    } else if b_shape.len() == 2 {
        if b_shape[0] != m {
            return Err(Error::ShapeMismatch {
                expected: vec![m, b_shape[1]],
                got: b_shape.to_vec(),
            });
        }
        (b_shape[1], false)
    } else {
        return Err(Error::Internal(format!(
            "lstsq requires b to be 1D or 2D, got {}D with shape {:?}",
            b_shape.len(),
            b_shape
        )));
    };

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
    let b_mat = if b_is_vector {
        b.reshape(&[m, 1])?.contiguous()
    } else {
        b.contiguous()
    };

    // Q^T @ B gives [m, num_rhs]
    let qtb = client.matmul(&q_t, &b_mat)?;

    let r_buffer = get_buffer(qr.r.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get R buffer".to_string()))?;

    // Allocate output X [n, num_rhs] or [n] for vector
    let x_total_size = n * num_rhs * dtype.size_in_bytes();
    let x_out_ptr = client.allocator().allocate(x_total_size);
    let x_out_buffer = get_buffer(x_out_ptr)
        .ok_or_else(|| Error::Internal("Failed to get x_out buffer".to_string()))?;

    if b_is_vector {
        // Single RHS: solve directly
        // Get first n elements of Q^T @ b using GPU-side slicing
        let qtb_flat = qtb.reshape(&[m])?;
        let qtb_n = qtb_flat.narrow(0, 0, n)?.contiguous();
        let qtb_buffer = get_buffer(qtb_n.storage().ptr())
            .ok_or_else(|| Error::Internal("Failed to get qtb buffer".to_string()))?;

        let params: [u32; 1] = [n as u32];
        let params_buffer = client.create_uniform_buffer("backward_params", 4);
        client.write_buffer(&params_buffer, &params);

        kernels::launch_backward_sub(
            client.pipeline_cache(),
            &client.queue,
            &r_buffer,
            &qtb_buffer,
            &x_out_buffer,
            &params_buffer,
            dtype,
        )?;
    } else {
        // Multi-RHS: solve for each column
        let qtb_contig = qtb.contiguous();
        let qtb_buffer = get_buffer(qtb_contig.storage().ptr())
            .ok_or_else(|| Error::Internal("Failed to get qtb buffer".to_string()))?;

        let col_size = n * dtype.size_in_bytes();
        let col_ptr = client.allocator().allocate(col_size);
        let col_buffer = get_buffer(col_ptr)
            .ok_or_else(|| Error::Internal("Failed to get col buffer".to_string()))?;
        let x_col_ptr = client.allocator().allocate(col_size);
        let x_col_buffer = get_buffer(x_col_ptr)
            .ok_or_else(|| Error::Internal("Failed to get x_col buffer".to_string()))?;

        for rhs in 0..num_rhs {
            // Extract first n elements of column 'rhs' from qtb [m, num_rhs]
            // We need to extract column and then take first n elements
            let extract_params: [u32; 4] = [n as u32, num_rhs as u32, rhs as u32, 0];
            let extract_params_buffer = client.create_uniform_buffer("extract_params", 16);
            client.write_buffer(&extract_params_buffer, &extract_params);
            kernels::launch_extract_column(
                client.pipeline_cache(),
                &client.queue,
                &qtb_buffer,
                &col_buffer,
                &extract_params_buffer,
                n, // only extract first n elements
                dtype,
            )?;

            // Solve R @ x_col = qtb_col[:n]
            let backward_params: [u32; 1] = [n as u32];
            let backward_params_buffer = client.create_uniform_buffer("backward_params", 4);
            client.write_buffer(&backward_params_buffer, &backward_params);
            kernels::launch_backward_sub(
                client.pipeline_cache(),
                &client.queue,
                &r_buffer,
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
