//! Banded linear system solver for WebGPU runtime
//!
//! Uses native WGSL compute shaders - NO CPU fallback, NO GPU↔CPU transfers.
//! F32 only (WGSL limitation).

use wgpu::CommandEncoderDescriptor;

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use crate::algorithm::linalg::{validate_linalg_dtype, validate_matrix_2d};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{Allocator, RuntimeClient};
use crate::tensor::Tensor;

/// Validate banded system inputs and return (n, nrhs).
fn validate_banded(
    ab_shape: &[usize],
    b_shape: &[usize],
    kl: usize,
    ku: usize,
) -> Result<(usize, usize)> {
    let (ab_rows, n) = validate_matrix_2d(ab_shape)?;
    let expected_rows = kl + ku + 1;
    if ab_rows != expected_rows {
        return Err(Error::ShapeMismatch {
            expected: vec![expected_rows, n],
            got: ab_shape.to_vec(),
        });
    }
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "ab",
            reason: "banded system size n must be > 0".to_string(),
        });
    }
    let nrhs = match b_shape.len() {
        1 => {
            if b_shape[0] != n {
                return Err(Error::ShapeMismatch {
                    expected: vec![n],
                    got: b_shape.to_vec(),
                });
            }
            1
        }
        2 => {
            if b_shape[0] != n {
                return Err(Error::ShapeMismatch {
                    expected: vec![n, b_shape[1]],
                    got: b_shape.to_vec(),
                });
            }
            b_shape[1]
        }
        _ => {
            return Err(Error::InvalidArgument {
                arg: "b",
                reason: format!("expected 1D or 2D tensor, got {}D", b_shape.len()),
            });
        }
    };
    Ok((n, nrhs))
}

pub fn solve_banded_impl(
    client: &WgpuClient,
    ab: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    kl: usize,
    ku: usize,
) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(ab.dtype())?;
    if ab.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: ab.dtype(),
            rhs: b.dtype(),
        });
    }
    let dtype = ab.dtype();
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "solve_banded (WebGPU: F32 only)",
        });
    }

    let (n, nrhs) = validate_banded(ab.shape(), b.shape(), kl, ku)?;
    let device = client.device();
    let elem_size = dtype.size_in_bytes();
    let col_size = n * elem_size;
    let b_is_1d = b.ndim() == 1;

    // Make inputs contiguous
    let ab_contig = ab.contiguous();
    let b_contig = b.contiguous();

    let ab_buffer = get_buffer(ab_contig.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get ab buffer".to_string()))?;
    let b_buffer = get_buffer(b_contig.storage().ptr())
        .ok_or_else(|| Error::Internal("Failed to get b buffer".to_string()))?;

    // Allocate output buffer for all RHS columns stored contiguously
    let x_total_size = n * nrhs * elem_size;
    let x_out_ptr = client.allocator().allocate(x_total_size);
    let x_out_buffer = get_buffer(x_out_ptr)
        .ok_or_else(|| Error::Internal("Failed to get x_out buffer".to_string()))?;

    let is_tridiagonal = kl == 1 && ku == 1;

    if nrhs == 1 {
        // Single RHS: solve directly into output
        if is_tridiagonal {
            // Thomas algorithm - needs: ab, b (copy to x as working rhs), x (output)
            // We need a copy of b as the shader modifies it in-place
            let b_copy_ptr = client.allocator().allocate(col_size);
            let b_copy_buffer = get_buffer(b_copy_ptr)
                .ok_or_else(|| Error::Internal("Failed to get b_copy buffer".to_string()))?;

            // Copy b to b_copy
            {
                let mut encoder =
                    client
                        .wgpu_device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("copy_b_for_thomas"),
                        });
                encoder.copy_buffer_to_buffer(&b_buffer, 0, &b_copy_buffer, 0, col_size as u64);
                client.queue.submit(std::iter::once(encoder.finish()));
            }

            let params: [u32; 2] = [n as u32, kl as u32]; // ThomasParams: n, ku (ku=kl=1)
            let params_buffer = client.create_uniform_buffer("thomas_params", 8);
            client.write_buffer(&params_buffer, &params);

            kernels::launch_thomas_solve(
                client.pipeline_cache(),
                &client.queue,
                &ab_buffer,
                &b_copy_buffer,
                &x_out_buffer,
                &params_buffer,
                dtype,
            )?;

            client.allocator().deallocate(b_copy_ptr, col_size);
        } else {
            // General banded LU
            let band_rows = kl + ku + 1;
            let work_rows = 2 * kl + ku + 1;
            let work_size = work_rows * n * elem_size;
            let work_ptr = client.allocator().allocate(work_size);
            let work_buffer = get_buffer(work_ptr)
                .ok_or_else(|| Error::Internal("Failed to get work buffer".to_string()))?;

            // Copy b to x_out (shader uses x as in-place rhs)
            {
                let mut encoder =
                    client
                        .wgpu_device
                        .create_command_encoder(&CommandEncoderDescriptor {
                            label: Some("copy_b_for_banded_lu"),
                        });
                encoder.copy_buffer_to_buffer(&b_buffer, 0, &x_out_buffer, 0, col_size as u64);
                client.queue.submit(std::iter::once(encoder.finish()));
            }

            let params: [u32; 4] = [n as u32, kl as u32, ku as u32, band_rows as u32];
            let params_buffer = client.create_uniform_buffer("banded_lu_params", 16);
            client.write_buffer(&params_buffer, &params);

            kernels::launch_banded_lu_solve(
                client.pipeline_cache(),
                &client.queue,
                &ab_buffer,
                &b_buffer,
                &x_out_buffer,
                &work_buffer,
                &params_buffer,
                dtype,
            )?;

            client.allocator().deallocate(work_ptr, work_size);
        }
    } else {
        // Multi-RHS: solve for each column
        let x_col_ptr = client.allocator().allocate(col_size);
        let x_col_buffer = get_buffer(x_col_ptr)
            .ok_or_else(|| Error::Internal("Failed to get x_col buffer".to_string()))?;

        let b_col_ptr = client.allocator().allocate(col_size);
        let b_col_buffer = get_buffer(b_col_ptr)
            .ok_or_else(|| Error::Internal("Failed to get b_col buffer".to_string()))?;

        // Allocate work buffer once for general banded (reused across RHS)
        let work_rows = 2 * kl + ku + 1;
        let work_size = work_rows * n * elem_size;
        let work_ptr = if !is_tridiagonal {
            let ptr = client.allocator().allocate(work_size);
            Some(ptr)
        } else {
            None
        };

        for rhs in 0..nrhs {
            // Extract column rhs from B [n, nrhs] into b_col_buffer
            let extract_params: [u32; 4] = [n as u32, nrhs as u32, rhs as u32, 0];
            let extract_params_buffer = client.create_uniform_buffer("extract_params", 16);
            client.write_buffer(&extract_params_buffer, &extract_params);
            kernels::launch_extract_column(
                client.pipeline_cache(),
                &client.queue,
                &b_buffer,
                &b_col_buffer,
                &extract_params_buffer,
                n,
                dtype,
            )?;

            if is_tridiagonal {
                let params: [u32; 2] = [n as u32, ku as u32];
                let params_buffer = client.create_uniform_buffer("thomas_params", 8);
                client.write_buffer(&params_buffer, &params);

                kernels::launch_thomas_solve(
                    client.pipeline_cache(),
                    &client.queue,
                    &ab_buffer,
                    &b_col_buffer,
                    &x_col_buffer,
                    &params_buffer,
                    dtype,
                )?;
            } else {
                let work_buffer = get_buffer(work_ptr.unwrap())
                    .ok_or_else(|| Error::Internal("Failed to get work buffer".to_string()))?;

                let band_rows = kl + ku + 1;
                let params: [u32; 4] = [n as u32, kl as u32, ku as u32, band_rows as u32];
                let params_buffer = client.create_uniform_buffer("banded_lu_params", 16);
                client.write_buffer(&params_buffer, &params);

                kernels::launch_banded_lu_solve(
                    client.pipeline_cache(),
                    &client.queue,
                    &ab_buffer,
                    &b_col_buffer,
                    &x_col_buffer,
                    &work_buffer,
                    &params_buffer,
                    dtype,
                )?;
            }

            // Copy x_col to the appropriate column in x_out
            let x_col_offset = rhs * col_size;
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

        client.allocator().deallocate(x_col_ptr, col_size);
        client.allocator().deallocate(b_col_ptr, col_size);
        if let Some(ptr) = work_ptr {
            client.allocator().deallocate(ptr, work_size);
        }
    }

    client.synchronize();

    // Create output tensor
    if b_is_1d {
        let x = unsafe { WgpuClient::tensor_from_raw(x_out_ptr, &[n], dtype, device) };
        Ok(x)
    } else {
        // Results stored as [nrhs, n] in memory (each column contiguous)
        let x_col_major =
            unsafe { WgpuClient::tensor_from_raw(x_out_ptr, &[nrhs, n], dtype, device) };
        let x = x_col_major.transpose(0, 1)?;
        Ok(x.contiguous())
    }
}
