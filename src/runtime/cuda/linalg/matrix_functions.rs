//! Matrix function implementations for CUDA (expm, logm, sqrtm)
//!
//! # Architecture: Full GPU Implementation
//!
//! This module runs all matrix function operations entirely on GPU:
//!
//! 1. **Schur decomposition: A = Z @ T @ Z^T** (O(n³)) → GPU
//! 2. **Matrix function on quasi-triangular T: f(T)** (O(n²)) → GPU kernels
//! 3. **Reconstruction: Z @ f(T) @ Z^T** (O(n³)) → GPU
//!
//! ## GPU Kernels
//!
//! - `diagonal_exp/log/sqrt` - Apply function to 1x1 and 2x2 diagonal blocks
//! - `parlett_column` - Compute off-diagonal elements via Parlett's recurrence
//! - `validate_eigenvalues` - Check for problematic eigenvalues
//!
//! ## Exception: funm
//!
//! The general `funm` function still uses CPU for Parlett recurrence because
//! it requires applying an arbitrary user-provided function that cannot be
//! compiled to GPU code.

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::super::kernels::linalg_launchers::{
    compute_schur_func_gpu, launch_validate_eigenvalues,
};
use crate::algorithm::linalg::{
    LinearAlgebraAlgorithms, matrix_functions_core, validate_linalg_dtype, validate_square_matrix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    BinaryOps, LinalgOps, MatmulOps, ReduceOps, ScalarOps, SortingOps, StatisticalOps, TensorOps,
    UnaryOps, UtilityOps,
};
use crate::runtime::{Allocator, RuntimeClient};
use crate::tensor::Tensor;

/// Get the GPU buffer pointer from a tensor.
fn get_tensor_ptr(tensor: &Tensor<CudaRuntime>) -> u64 {
    tensor.storage().ptr()
}

/// Matrix exponential using Schur decomposition - fully on GPU.
pub fn expm_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device));
    }

    if n == 1 {
        // Single element - use GPU unary exp
        return client.exp(a);
    }

    // Compute Schur decomposition: A = Z @ T @ Z^T (GPU)
    let schur = client.schur_decompose(a)?;

    // Allocate output for f(T) on GPU
    let f_t = Tensor::<CudaRuntime>::zeros(&[n, n], dtype, device);

    // Compute exp(T) entirely on GPU
    let t_ptr = get_tensor_ptr(&schur.t);
    let f_ptr = get_tensor_ptr(&f_t);

    unsafe {
        compute_schur_func_gpu(
            client.context(),
            client.stream(),
            client.device().index,
            dtype,
            t_ptr,
            f_ptr,
            n,
            "exp",
        )?;
    }

    // Reconstruct: exp(A) = Z @ exp(T) @ Z^T on GPU
    let temp = client.matmul(&schur.z, &f_t)?;
    let z_t = schur.z.transpose(0, 1)?;
    client.matmul(&temp, &z_t)
}

/// Matrix logarithm using Schur decomposition - fully on GPU.
pub fn logm_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device));
    }

    if n == 1 {
        // Single element - use GPU unary log
        // But we need to check for non-positive value first
        let data: Vec<f64> = a.to_vec();
        let val = data[0];
        if val <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "a",
                reason: "logm requires matrix with no non-positive real eigenvalues".to_string(),
            });
        }
        return client.log(a);
    }

    let eps = match dtype {
        DType::F32 => f32::EPSILON as f64,
        DType::F64 => f64::EPSILON,
        _ => f64::EPSILON,
    };

    // Compute Schur decomposition: A = Z @ T @ Z^T (GPU)
    let schur = client.schur_decompose(a)?;

    // Validate eigenvalues on GPU
    let result_buffer = client.allocator.allocate(2 * dtype.size_in_bytes());
    // Zero-initialize the result buffer
    let zero_data: [f64; 2] = [0.0, 0.0];
    unsafe {
        cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
            result_buffer,
            zero_data.as_ptr() as *const std::ffi::c_void,
            2 * std::mem::size_of::<f64>(),
            client.stream().cu_stream(),
        );
    }

    unsafe {
        launch_validate_eigenvalues(
            client.context(),
            client.stream(),
            client.device().index,
            dtype,
            get_tensor_ptr(&schur.t),
            result_buffer,
            n,
            eps,
            "log",
        )?;
    }

    // Synchronize and check result
    client
        .stream()
        .synchronize()
        .map_err(|e| Error::Internal(format!("CUDA stream synchronize failed: {:?}", e)))?;

    let mut result_data: [f64; 2] = [0.0, 0.0];
    unsafe {
        cudarc::driver::sys::cuMemcpyDtoH_v2(
            result_data.as_mut_ptr() as *mut std::ffi::c_void,
            result_buffer,
            2 * std::mem::size_of::<f64>(),
        );
    }
    client
        .allocator
        .deallocate(result_buffer, 2 * dtype.size_in_bytes());

    if result_data[0] > 0.5 {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: format!(
                "logm requires matrix with no non-positive real eigenvalues, found {}",
                result_data[1]
            ),
        });
    }

    // Allocate output for f(T) on GPU
    let f_t = Tensor::<CudaRuntime>::zeros(&[n, n], dtype, device);

    // Compute log(T) entirely on GPU
    unsafe {
        compute_schur_func_gpu(
            client.context(),
            client.stream(),
            client.device().index,
            dtype,
            get_tensor_ptr(&schur.t),
            get_tensor_ptr(&f_t),
            n,
            "log",
        )?;
    }

    // Reconstruct: log(A) = Z @ log(T) @ Z^T on GPU
    let temp = client.matmul(&schur.z, &f_t)?;
    let z_t = schur.z.transpose(0, 1)?;
    client.matmul(&temp, &z_t)
}

/// Matrix square root using Denman-Beavers iteration - fully on GPU.
pub fn sqrtm_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device));
    }

    if n == 1 {
        let data: Vec<f64> = a.to_vec();
        let val = data[0];
        if val < 0.0 {
            return Err(Error::InvalidArgument {
                arg: "a",
                reason: "sqrtm requires matrix with no negative real eigenvalues".to_string(),
            });
        }
        return client.sqrt(a);
    }

    let eps = match dtype {
        DType::F32 => f32::EPSILON as f64,
        DType::F64 => f64::EPSILON,
        _ => f64::EPSILON,
    };

    // Check for negative real eigenvalues using Schur decomposition
    let schur = client.schur_decompose(a)?;

    // Validate eigenvalues on GPU
    let result_buffer = client.allocator.allocate(2 * dtype.size_in_bytes());
    // Zero-initialize the result buffer
    let zero_data: [f64; 2] = [0.0, 0.0];
    unsafe {
        cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
            result_buffer,
            zero_data.as_ptr() as *const std::ffi::c_void,
            2 * std::mem::size_of::<f64>(),
            client.stream().cu_stream(),
        );
    }

    unsafe {
        launch_validate_eigenvalues(
            client.context(),
            client.stream(),
            client.device().index,
            dtype,
            get_tensor_ptr(&schur.t),
            result_buffer,
            n,
            eps,
            "sqrt",
        )?;
    }

    // Synchronize and check result
    client
        .stream()
        .synchronize()
        .map_err(|e| Error::Internal(format!("CUDA stream synchronize failed: {:?}", e)))?;

    let mut result_data: [f64; 2] = [0.0, 0.0];
    unsafe {
        cudarc::driver::sys::cuMemcpyDtoH_v2(
            result_data.as_mut_ptr() as *mut std::ffi::c_void,
            result_buffer,
            2 * std::mem::size_of::<f64>(),
        );
    }
    client
        .allocator
        .deallocate(result_buffer, 2 * dtype.size_in_bytes());

    if result_data[0] > 0.5 {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: format!(
                "sqrtm requires matrix with no negative real eigenvalues, found {}",
                result_data[1]
            ),
        });
    }

    // Denman-Beavers iteration on GPU
    let mut y = a.clone();
    let mut z = client.eye(n, None, dtype)?;

    let max_iter = 50;
    let tol = eps * (n as f64);

    for _iter in 0..max_iter {
        // Compute Y_inv and Z_inv
        let y_inv = match LinearAlgebraAlgorithms::inverse(client, &y) {
            Ok(inv) => inv,
            Err(_) => {
                return Err(Error::Internal(
                    "sqrtm: matrix inversion failed during iteration".to_string(),
                ));
            }
        };

        let z_inv = match LinearAlgebraAlgorithms::inverse(client, &z) {
            Ok(inv) => inv,
            Err(_) => {
                return Err(Error::Internal(
                    "sqrtm: matrix inversion failed during iteration".to_string(),
                ));
            }
        };

        // Y_new = (Y + Z_inv) / 2
        let y_plus_zinv = client.add(&y, &z_inv)?;
        let y_new = client.div_scalar(&y_plus_zinv, 2.0)?;

        // Z_new = (Z + Y_inv) / 2
        let z_plus_yinv = client.add(&z, &y_inv)?;
        let z_new = client.div_scalar(&z_plus_yinv, 2.0)?;

        // Check convergence using GPU reduce operations
        let diff = client.sub(&y_new, &y)?;
        let diff_sq = client.mul(&diff, &diff)?;
        let diff_sum = client.sum(&diff_sq, &[], false)?;

        let y_sq = client.mul(&y, &y)?;
        let y_sum = client.sum(&y_sq, &[], false)?;

        // Extract scalar results (small transfer, unavoidable for convergence check)
        let diff_norm: f64 = {
            let sum_vec: Vec<f64> = diff_sum.to_vec();
            sum_vec[0].sqrt()
        };

        let y_norm: f64 = {
            let sum_vec: Vec<f64> = y_sum.to_vec();
            sum_vec[0].sqrt().max(1.0)
        };

        y = y_new;
        z = z_new;

        if diff_norm / y_norm < tol {
            break;
        }
    }

    Ok(y)
}

/// Matrix sign function using Newton iteration - fully on GPU.
pub fn signm_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    let n = crate::algorithm::linalg::validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device));
    }

    if n == 1 {
        let data: Vec<f64> = a.to_vec();
        let val = data[0];
        if val.abs() < f64::EPSILON {
            return Err(Error::InvalidArgument {
                arg: "a",
                reason: "signm requires matrix with no zero eigenvalues".to_string(),
            });
        }
        let sign_val = if val > 0.0 { 1.0 } else { -1.0 };
        return Ok(Tensor::<CudaRuntime>::full_scalar(
            &[1, 1],
            dtype,
            sign_val,
            device,
        ));
    }

    let eps = match dtype {
        DType::F32 => f32::EPSILON as f64,
        DType::F64 => f64::EPSILON,
        _ => f64::EPSILON,
    };

    // Newton iteration: X_{k+1} = (X_k + X_k^{-1}) / 2
    let mut x = a.clone();
    let max_iter = 100;
    let tol = eps * (n as f64).sqrt();

    for _iter in 0..max_iter {
        // Compute X^{-1}
        let x_inv = match LinearAlgebraAlgorithms::inverse(client, &x) {
            Ok(inv) => inv,
            Err(_) => {
                return Err(Error::Internal(
                    "signm: matrix became singular during iteration".to_string(),
                ));
            }
        };

        // X_new = (X + X^{-1}) / 2
        let x_plus_inv = client.add(&x, &x_inv)?;
        let x_new = client.div_scalar(&x_plus_inv, 2.0)?;

        // Check convergence using GPU reduce
        let diff = client.sub(&x_new, &x)?;
        let diff_sq = client.mul(&diff, &diff)?;
        let diff_sum = client.sum(&diff_sq, &[], false)?;

        // Extract scalar result (small transfer, unavoidable for convergence check)
        let diff_norm: f64 = {
            let sum_vec: Vec<f64> = diff_sum.to_vec();
            sum_vec[0].sqrt()
        };

        x = x_new;

        if diff_norm < tol {
            break;
        }
    }

    Ok(x)
}

/// Fractional matrix power: A^p using log and exp
pub fn fractional_matrix_power_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    p: f64,
) -> Result<Tensor<CudaRuntime>> {
    let n = crate::algorithm::linalg::validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device));
    }

    // p = 0: Return identity
    if p.abs() < f64::EPSILON {
        return client.eye(n, None, dtype);
    }

    // p = 1: Return A
    if (p - 1.0).abs() < f64::EPSILON {
        return Ok(a.clone());
    }

    if n == 1 {
        let data: Vec<f64> = a.to_vec();
        let val = data[0];
        if val <= 0.0 && p.fract() != 0.0 {
            return Err(Error::InvalidArgument {
                arg: "a",
                reason: "fractional_matrix_power requires positive eigenvalues for non-integer p"
                    .to_string(),
            });
        }
        let result = val.powf(p);
        return Ok(Tensor::<CudaRuntime>::full_scalar(
            &[1, 1],
            dtype,
            result,
            device,
        ));
    }

    // p = -1: Return inverse
    if (p + 1.0).abs() < f64::EPSILON {
        return LinearAlgebraAlgorithms::inverse(client, a);
    }

    // p = 0.5: Use sqrtm (more accurate)
    if (p - 0.5).abs() < f64::EPSILON {
        return sqrtm_impl(client, a);
    }

    // Integer powers: use repeated squaring
    if p.fract() == 0.0 && p.abs() < 100.0 {
        return integer_matrix_power_gpu(client, a, n, p as i64);
    }

    // General case: A^p = exp(p * log(A))
    let log_a = logm_impl(client, a)?;
    let p_log_a = client.mul_scalar(&log_a, p)?;
    expm_impl(client, &p_log_a)
}

/// Integer matrix power using repeated squaring on GPU
fn integer_matrix_power_gpu(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    n: usize,
    p: i64,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();

    if p == 0 {
        return client.eye(n, None, dtype);
    }

    // Handle negative powers
    let (mut base, mut exp) = if p < 0 {
        let inv = LinearAlgebraAlgorithms::inverse(client, a)?;
        (inv, (-p) as u64)
    } else {
        (a.clone(), p as u64)
    };

    // Repeated squaring
    let mut result = client.eye(n, None, dtype)?;

    while exp > 0 {
        if exp & 1 == 1 {
            result = client.matmul(&result, &base)?;
        }
        base = client.matmul(&base, &base)?;
        exp >>= 1;
    }

    Ok(result)
}

/// General matrix function f(A) using Schur-Parlett algorithm.
///
/// Note: This function still uses CPU for the Parlett recurrence because it
/// requires applying an arbitrary user-provided function that cannot be
/// compiled to GPU code. For exp, log, sqrt, use the dedicated GPU implementations.
pub fn funm_impl<F>(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    f: F,
) -> Result<Tensor<CudaRuntime>>
where
    F: Fn(f64) -> f64,
{
    let n = crate::algorithm::linalg::validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device));
    }

    if n == 1 {
        let data: Vec<f64> = a.to_vec();
        let val = data[0];
        let result = f(val);
        if result.is_nan() || result.is_infinite() {
            return Err(Error::InvalidArgument {
                arg: "f",
                reason: "function returned NaN or Inf for eigenvalue".to_string(),
            });
        }
        return Ok(Tensor::<CudaRuntime>::full_scalar(
            &[1, 1],
            dtype,
            result,
            device,
        ));
    }

    // Compute Schur decomposition: A = Z @ T @ Z^T
    let schur = client.schur_decompose(a)?;
    let t_data: Vec<f64> = schur.t.to_vec();
    let z_data: Vec<f64> = schur.z.to_vec();

    // Compute f(T) using shared algorithm (CPU - arbitrary function cannot be GPU-ified)
    let f_t = matrix_functions_core::funm_quasi_triangular_f64(&t_data, n, &f)?;

    // Reconstruct: f(A) = Z @ f(T) @ Z^T on GPU
    let f_t_tensor = Tensor::<CudaRuntime>::from_slice(&f_t, &[n, n], device);
    let z_tensor = Tensor::<CudaRuntime>::from_slice(&z_data, &[n, n], device);

    let temp = client.matmul(&z_tensor, &f_t_tensor)?;
    let z_t = z_tensor.transpose(0, 1)?;
    client.matmul(&temp, &z_t)
}
