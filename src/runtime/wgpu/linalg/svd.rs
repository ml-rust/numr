//! SVD and related operations (pinverse, cond).

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use super::helpers::get_buffer_or_err;
use crate::algorithm::linalg::{SvdDecomposition, validate_linalg_dtype, validate_matrix_2d};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{LinalgOps, MatmulOps, ReduceOps};
use crate::runtime::{Allocator, Runtime, RuntimeClient};
use crate::tensor::Tensor;

pub fn svd_decompose(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
) -> Result<SvdDecomposition<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU svd_decompose (only F32 supported)",
        });
    }

    // Handle transpose for m < n case
    let transposed = m < n;
    let (work_m, work_n) = if transposed { (n, m) } else { (m, n) };
    let k = work_m.min(work_n);

    // Get input data
    let a_data: Vec<f32> = a.to_vec();

    // If transposed, compute A^T
    let work_data = if transposed {
        let mut transposed_data = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                transposed_data[j * m + i] = a_data[i * n + j];
            }
        }
        transposed_data
    } else {
        a_data
    };

    // Allocate buffers for SVD computation
    let b_size = work_m * work_n * dtype.size_in_bytes();
    let b_ptr = client.allocator().allocate(b_size);
    let b_buffer = get_buffer_or_err!(b_ptr, "B (working matrix)");

    let v_size = work_n * work_n * dtype.size_in_bytes();
    let v_ptr = client.allocator().allocate(v_size);
    let v_buffer = get_buffer_or_err!(v_ptr, "V (right singular vectors)");

    let s_size = work_n * dtype.size_in_bytes();
    let s_ptr = client.allocator().allocate(s_size);
    let s_buffer = get_buffer_or_err!(s_ptr, "S (singular values)");

    let converged_flag_size = std::mem::size_of::<i32>();
    let converged_flag_ptr = client.allocator().allocate(converged_flag_size);
    let converged_flag_buffer = get_buffer_or_err!(converged_flag_ptr, "SVD convergence flag");

    // Copy working data to B buffer
    WgpuRuntime::copy_to_device(bytemuck::cast_slice(&work_data), b_ptr, device);

    // Zero-initialize converged flag
    let zero_i32: [i32; 1] = [0];
    client.write_buffer(&converged_flag_buffer, &zero_i32);

    // Create params buffer
    let params: [u32; 2] = [work_m as u32, work_n as u32];
    let params_buffer = client.create_uniform_buffer("svd_params", 8);
    client.write_buffer(&params_buffer, &params);

    // Launch SVD kernel
    kernels::launch_svd_jacobi(
        client.pipeline_cache(),
        &client.queue,
        &b_buffer,
        &v_buffer,
        &s_buffer,
        &converged_flag_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    // Read back converged flag
    let staging = client.create_staging_buffer("svd_converged_staging", 4);
    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("svd_converged_copy"),
        });
    encoder.copy_buffer_to_buffer(&converged_flag_buffer, 0, &staging, 0, 4);
    client.submit_and_wait(encoder);

    let mut converged_val = [0i32; 1];
    client.read_buffer(&staging, &mut converged_val);

    client
        .allocator()
        .deallocate(converged_flag_ptr, converged_flag_size);

    if converged_val[0] != 0 {
        client.allocator().deallocate(b_ptr, b_size);
        client.allocator().deallocate(v_ptr, v_size);
        client.allocator().deallocate(s_ptr, s_size);
        return Err(Error::Internal(
            "SVD did not converge within maximum iterations".to_string(),
        ));
    }

    // Read results back
    let b_staging = client.create_staging_buffer("svd_b_staging", b_size as u64);
    let v_staging = client.create_staging_buffer("svd_v_staging", v_size as u64);

    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("svd_results_copy"),
        });
    encoder.copy_buffer_to_buffer(&b_buffer, 0, &b_staging, 0, b_size as u64);
    encoder.copy_buffer_to_buffer(&v_buffer, 0, &v_staging, 0, v_size as u64);
    client.submit_and_wait(encoder);

    let mut b_data = vec![0.0f32; work_m * work_n];
    let mut v_data = vec![0.0f32; work_n * work_n];
    client.read_buffer(&b_staging, &mut b_data);
    client.read_buffer(&v_staging, &mut v_data);

    // Deallocate working buffers
    client.allocator().deallocate(b_ptr, b_size);
    client.allocator().deallocate(v_ptr, v_size);

    // Handle transpose
    let (u_final, vt_final) = if transposed {
        let mut u_final = vec![0.0f32; m * k];
        for i in 0..m {
            for j in 0..k {
                u_final[i * k + j] = v_data[i * work_n + j];
            }
        }

        let mut vt_final = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                vt_final[i * n + j] = b_data[j * work_n + i];
            }
        }

        (u_final, vt_final)
    } else {
        let mut u_final = vec![0.0f32; m * k];
        for i in 0..m {
            for j in 0..k {
                u_final[i * k + j] = b_data[i * n + j];
            }
        }

        let mut vt_final = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                vt_final[i * n + j] = v_data[j * n + i];
            }
        }

        (u_final, vt_final)
    };

    // Allocate final output tensors on GPU
    let u_size = u_final.len() * dtype.size_in_bytes();
    let u_ptr = client.allocator().allocate(u_size);
    WgpuRuntime::copy_to_device(bytemuck::cast_slice(&u_final), u_ptr, device);

    let vt_size = vt_final.len() * dtype.size_in_bytes();
    let vt_ptr = client.allocator().allocate(vt_size);
    WgpuRuntime::copy_to_device(bytemuck::cast_slice(&vt_final), vt_ptr, device);

    // Create output tensors
    let u = unsafe { WgpuClient::tensor_from_raw(u_ptr, &[m, k], dtype, device) };
    let s = unsafe { WgpuClient::tensor_from_raw(s_ptr, &[k], dtype, device) };
    let vt = unsafe { WgpuClient::tensor_from_raw(vt_ptr, &[k, n], dtype, device) };

    Ok(SvdDecomposition { u, s, vt })
}

pub fn pinverse(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    rcond: Option<f64>,
) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "pinverse",
        });
    }

    // Handle empty matrix
    if m == 0 || n == 0 {
        let out_ptr = client.allocator().allocate(0);
        return Ok(unsafe { WgpuClient::tensor_from_raw(out_ptr, &[n, m], dtype, device) });
    }

    // Compute SVD
    let svd = svd_decompose(client, a)?;

    let k = m.min(n);

    // Compute max singular value on GPU
    let max_sv_tensor = client.max(&svd.s, &[0], false)?;
    let max_sv = max_sv_tensor.to_vec::<f32>()[0] as f64;

    // Determine cutoff threshold
    let default_rcond = (m.max(n) as f64) * (f32::EPSILON as f64);
    let rcond_val = rcond.unwrap_or(default_rcond);
    let cutoff = (rcond_val * max_sv) as f32;

    // Compute S_inv (still needs CPU for conditional - TODO: GPU kernel)
    let s_data: Vec<f32> = svd.s.to_vec();
    let s_inv_data: Vec<f32> = s_data
        .iter()
        .map(|&s| if s > cutoff { 1.0 / s } else { 0.0 })
        .collect();

    // Create S_inv diagonal matrix on GPU
    let s_inv_diag = Tensor::<WgpuRuntime>::from_slice(&s_inv_data, &[k], device);
    let s_inv_mat = client.diagflat(&s_inv_diag)?;

    // Compute A^+ = V @ S_inv @ U^T
    let v = svd.vt.transpose(0, 1)?.contiguous();
    let ut = svd.u.transpose(0, 1)?.contiguous();

    let v_sinv = client.matmul(&v, &s_inv_mat)?;
    let pinv = client.matmul(&v_sinv, &ut)?;

    Ok(pinv)
}

pub fn cond(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op: "cond" });
    }

    // Handle empty matrix
    if m == 0 || n == 0 {
        return Ok(Tensor::<WgpuRuntime>::from_slice(
            &[f32::INFINITY],
            &[],
            device,
        ));
    }

    // Compute SVD
    let svd = svd_decompose(client, a)?;

    // Condition number = max(S) / min(S) using GPU reduce operations
    let max_sv_tensor = client.max(&svd.s, &[0], false)?;
    let min_sv_tensor = client.min(&svd.s, &[0], false)?;

    // Extract scalar results (small transfer - just 2 floats)
    let max_sv: f32 = max_sv_tensor.to_vec::<f32>()[0];
    let min_sv: f32 = min_sv_tensor.to_vec::<f32>()[0];

    let cond_val = if min_sv == 0.0 || !min_sv.is_finite() {
        f32::INFINITY
    } else {
        max_sv / min_sv
    };

    Ok(Tensor::<WgpuRuntime>::from_slice(&[cond_val], &[], device))
}
