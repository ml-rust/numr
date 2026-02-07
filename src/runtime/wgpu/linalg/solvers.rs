//! Linear system solvers for WebGPU backend.
//!
//! # Supported Data Types
//!
//! **F32 only.** WebGPU WGSL shaders do not natively support F64 operations.
//! All solver functions in this module will return `Error::UnsupportedDType`
//! for non-F32 inputs.
//!
//! For F64 linear algebra, use the CPU or CUDA backends instead.

use wgpu::CommandEncoderDescriptor;

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use crate::algorithm::linalg::{validate_linalg_dtype, validate_square_matrix};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{Allocator, RuntimeClient};
use crate::tensor::Tensor;

// Re-export from sub-modules
pub use super::lstsq::lstsq;
pub use super::triangular_solve::{solve_triangular_lower, solve_triangular_upper};

/// Solve linear system Ax = b using LU decomposition with partial pivoting.
/// Supports both single RHS (b is 1D vector) and multi-RHS (b is 2D matrix [n, nrhs])
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

    // Handle 1D or 2D b (like CPU version)
    let b_shape = b.shape();
    let (b_rows, num_rhs) = if b_shape.len() == 1 {
        (b_shape[0], 1)
    } else if b_shape.len() == 2 {
        (b_shape[0], b_shape[1])
    } else {
        return Err(Error::ShapeMismatch {
            expected: vec![n],
            got: b_shape.to_vec(),
        });
    };

    if b_rows != n {
        return Err(Error::ShapeMismatch {
            expected: vec![n],
            got: vec![b_rows],
        });
    }

    // Compute LU decomposition using decompositions module
    use super::decompositions::lu_decompose;
    let lu_result = lu_decompose(client, a)?;

    // Get LU and pivots buffers (both already on GPU, no transfers needed)
    let lu_buffer = get_buffer(lu_result.lu.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get lu buffer".to_string()))?;
    let pivots_buffer = get_buffer(lu_result.pivots.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get pivots buffer".to_string()))?;

    // Allocate temporary buffers for single column operations
    let col_size = n * dtype.size_in_bytes();
    let pb_ptr = client.allocator().allocate(col_size);
    let pb_buffer =
        get_buffer(pb_ptr).ok_or_else(|| Error::Internal("Failed to get pb buffer".to_string()))?;
    let y_ptr = client.allocator().allocate(col_size);
    let y_buffer =
        get_buffer(y_ptr).ok_or_else(|| Error::Internal("Failed to get y buffer".to_string()))?;
    let col_ptr = client.allocator().allocate(col_size);
    let col_buffer = get_buffer(col_ptr)
        .ok_or_else(|| Error::Internal("Failed to get col buffer".to_string()))?;

    // Get b buffer for GPU column extraction
    let b_contig = b.contiguous();
    let b_buffer = get_buffer(b_contig.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get b buffer".to_string()))?;

    // Allocate output buffer for all RHS (column-major: each solved column stored contiguously)
    let x_total_size = n * num_rhs * dtype.size_in_bytes();
    let x_out_ptr = client.allocator().allocate(x_total_size);
    let x_out_buffer = get_buffer(x_out_ptr)
        .ok_or_else(|| Error::Internal("Failed to get x_out buffer".to_string()))?;

    // Solve for each right-hand side using GPU kernels - NO CPU transfers
    for rhs in 0..num_rhs {
        // Extract column from b using GPU kernel
        if num_rhs == 1 {
            // For 1D case, just copy b directly to col_buffer
            let copy_params: [u32; 4] = [n as u32, 0, 0, 0];
            let copy_params_buffer = client.create_uniform_buffer("copy_params", 16);
            client.write_buffer(&copy_params_buffer, &copy_params);
            kernels::launch_matrix_copy(
                client.pipeline_cache(),
                &client.queue,
                &b_buffer,
                &col_buffer,
                &copy_params_buffer,
                n,
                dtype,
            )?;
        } else {
            // Extract column 'rhs' from b[n, num_rhs] into col_buffer
            // ExtractParams: m (rows), n_cols (columns), col (which column)
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
        }

        // Apply permutation: pb = P @ col
        let perm_params: [u32; 4] = [n as u32, 0, 0, 0];
        let perm_params_buffer = client.create_uniform_buffer("perm_params", 16);
        client.write_buffer(&perm_params_buffer, &perm_params);

        kernels::launch_apply_lu_permutation(
            client.pipeline_cache(),
            &client.queue,
            &col_buffer,
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

        // Backward substitution: Ux = y (result goes to x_col portion of output)
        // We write directly to the right column offset in x_out_buffer
        let backward_params: [u32; 4] = [n as u32, 0, 0, 0];
        let backward_params_buffer = client.create_uniform_buffer("backward_params", 16);
        client.write_buffer(&backward_params_buffer, &backward_params);

        // Calculate offset for this column in output buffer
        let x_col_offset = rhs * n * dtype.size_in_bytes();

        // Use a temp buffer for backward substitution then copy to output
        let x_col_ptr = client.allocator().allocate(col_size);
        let x_col_buffer = get_buffer(x_col_ptr)
            .ok_or_else(|| Error::Internal("Failed to get x_col buffer".to_string()))?;

        kernels::launch_backward_sub(
            client.pipeline_cache(),
            &client.queue,
            &lu_buffer,
            &y_buffer,
            &x_col_buffer,
            &backward_params_buffer,
            dtype,
        )?;

        // Copy x_col to the appropriate column in x_out using buffer-to-buffer copy
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

        client.allocator().deallocate(x_col_ptr, col_size);
    }

    client.synchronize();

    // Clean up temporary buffers
    client.allocator().deallocate(pb_ptr, col_size);
    client.allocator().deallocate(y_ptr, col_size);
    client.allocator().deallocate(col_ptr, col_size);

    // Create output tensor
    // Results are stored in column-major order (each solved column is contiguous)
    if b_shape.len() == 1 {
        // 1D case: just return as [n]
        let x = unsafe { WgpuClient::tensor_from_raw(x_out_ptr, &[n], dtype, device) };
        Ok(x)
    } else {
        // 2D case: stored as [num_rhs, n] in memory (column-major for the [n, num_rhs] result)
        // Create tensor as [num_rhs, n] then transpose to get [n, num_rhs] view
        let x_col_major =
            unsafe { WgpuClient::tensor_from_raw(x_out_ptr, &[num_rhs, n], dtype, device) };
        // Transpose to get [n, num_rhs] - this is a zero-copy view
        let x = x_col_major.transpose(0, 1)?;
        // Make contiguous to get proper row-major layout
        Ok(x.contiguous())
    }
}
