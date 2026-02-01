//! Matrix function implementations for CUDA (expm, logm, sqrtm)
//!
//! # Architecture: Hybrid GPU/CPU Approach
//!
//! This module uses a **hybrid approach** that leverages GPU for parallelizable
//! operations and CPU for inherently sequential operations:
//!
//! ## Why Hybrid?
//!
//! Matrix functions via Schur decomposition have three phases:
//!
//! 1. **Schur decomposition: A = Z @ T @ Z^T** (O(n³))
//!    - QR iteration with Householder reflections
//!    - Highly parallelizable → **GPU**
//!
//! 2. **Matrix function on quasi-triangular T: f(T)** (O(n²))
//!    - Parlett recurrence: sequential diagonal-by-diagonal
//!    - Inherently sequential → **CPU** (via `matrix_functions_core`)
//!
//! 3. **Reconstruction: Z @ f(T) @ Z^T** (O(n³))
//!    - Two matrix multiplications
//!    - Highly parallelizable → **GPU**
//!
//! ## Performance Characteristics
//!
//! For an n×n matrix:
//! - Steps 1 and 3 dominate (O(n³) each)
//! - Step 2 is O(n²) and sequential regardless of hardware
//! - Transfer overhead is O(n²), negligible compared to O(n³) compute
//!
//! This hybrid approach achieves near-optimal performance by running
//! parallelizable operations on GPU and avoiding GPU overhead for
//! operations that wouldn't benefit from it.
//!
//! ## Exception: sqrtm and signm
//!
//! These use iterative methods (Denman-Beavers, Newton) that are
//! implemented entirely on GPU since each iteration involves matrix
//! multiplications and inversions that benefit from GPU acceleration.

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use crate::algorithm::linalg::{
    LinearAlgebraAlgorithms, matrix_functions_core, validate_linalg_dtype, validate_square_matrix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    BinaryOps, LinalgOps, MatmulOps, ReduceOps, ScalarOps, SortingOps, StatisticalOps, TensorOps,
    UnaryOps, UtilityOps,
};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Matrix exponential using Schur decomposition
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
        let data: Vec<f64> = a.to_vec();
        let exp_val = data[0].exp();
        return Ok(Tensor::<CudaRuntime>::full_scalar(
            &[1, 1],
            dtype,
            exp_val,
            device,
        ));
    }

    // Compute Schur decomposition: A = Z @ T @ Z^T
    let schur = client.schur_decompose(a)?;

    // Transfer T to CPU for quasi-triangular exponential computation
    let t_data: Vec<f64> = schur.t.to_vec();
    let z_data: Vec<f64> = schur.z.to_vec();

    // Compute exp(T) for quasi-triangular T using shared algorithm
    let exp_t = matrix_functions_core::exp_quasi_triangular_f64(&t_data, n);

    // Reconstruct: exp(A) = Z @ exp(T) @ Z^T on GPU
    let exp_t_tensor = Tensor::<CudaRuntime>::from_slice(&exp_t, &[n, n], device);
    let z_tensor = Tensor::<CudaRuntime>::from_slice(&z_data, &[n, n], device);

    // temp = Z @ exp(T)
    let temp = client.matmul(&z_tensor, &exp_t_tensor)?;

    // Z^T
    let z_t = z_tensor.transpose(0, 1)?;

    // result = temp @ Z^T
    client.matmul(&temp, &z_t)
}

/// Matrix square root using Denman-Beavers iteration
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
        return Ok(Tensor::<CudaRuntime>::full_scalar(
            &[1, 1],
            dtype,
            val.sqrt(),
            device,
        ));
    }

    // Check for negative real eigenvalues using Schur decomposition
    let schur = client.schur_decompose(a)?;
    let t_data: Vec<f64> = schur.t.to_vec();
    let eps = if dtype == DType::F32 {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    };

    let mut i = 0;
    while i < n {
        if i + 1 < n && t_data[(i + 1) * n + i].abs() > eps {
            i += 2;
        } else {
            let eigenvalue = t_data[i * n + i];
            if eigenvalue < -eps {
                return Err(Error::InvalidArgument {
                    arg: "a",
                    reason: format!(
                        "sqrtm requires matrix with no negative real eigenvalues, found {}",
                        eigenvalue
                    ),
                });
            }
            i += 1;
        }
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

        // Check convergence: ||Y_new - Y|| / ||Y||
        let diff = client.sub(&y_new, &y)?;
        let diff_norm: f64 = {
            let diff_sq = client.mul(&diff, &diff)?;
            let sum = client.sum(&diff_sq, &[], false)?;
            let sum_vec: Vec<f64> = sum.to_vec();
            sum_vec[0].sqrt()
        };

        let y_norm: f64 = {
            let y_sq = client.mul(&y, &y)?;
            let sum = client.sum(&y_sq, &[], false)?;
            let sum_vec: Vec<f64> = sum.to_vec();
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

/// Matrix logarithm using Schur decomposition
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
        let data: Vec<f64> = a.to_vec();
        let val = data[0];
        if val <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "a",
                reason: "logm requires matrix with no non-positive real eigenvalues".to_string(),
            });
        }
        return Ok(Tensor::<CudaRuntime>::full_scalar(
            &[1, 1],
            dtype,
            val.ln(),
            device,
        ));
    }

    let eps = if dtype == DType::F32 {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    };

    // Compute Schur decomposition: A = Z @ T @ Z^T
    let schur = client.schur_decompose(a)?;
    let t_data: Vec<f64> = schur.t.to_vec();
    let z_data: Vec<f64> = schur.z.to_vec();

    // Check for non-positive real eigenvalues
    let mut i = 0;
    while i < n {
        if i + 1 < n && t_data[(i + 1) * n + i].abs() > eps {
            let a_val = t_data[i * n + i];
            let b_val = t_data[i * n + (i + 1)];
            let c_val = t_data[(i + 1) * n + i];
            if a_val <= eps && b_val * c_val >= -eps {
                return Err(Error::InvalidArgument {
                    arg: "a",
                    reason: "logm requires matrix with no non-positive real eigenvalues"
                        .to_string(),
                });
            }
            i += 2;
        } else {
            let eigenvalue = t_data[i * n + i];
            if eigenvalue <= eps {
                return Err(Error::InvalidArgument {
                    arg: "a",
                    reason: format!(
                        "logm requires matrix with no non-positive real eigenvalues, found {}",
                        eigenvalue
                    ),
                });
            }
            i += 1;
        }
    }

    // Compute log(T) using shared algorithm
    let log_t = matrix_functions_core::log_quasi_triangular_f64(&t_data, n, eps);

    // Reconstruct: log(A) = Z @ log(T) @ Z^T on GPU
    let log_t_tensor = Tensor::<CudaRuntime>::from_slice(&log_t, &[n, n], device);
    let z_tensor = Tensor::<CudaRuntime>::from_slice(&z_data, &[n, n], device);

    let temp = client.matmul(&z_tensor, &log_t_tensor)?;
    let z_t = z_tensor.transpose(0, 1)?;
    client.matmul(&temp, &z_t)
}

/// Matrix sign function using Newton iteration
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

    let eps = if dtype == DType::F32 {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
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

        // Check convergence
        let diff = client.sub(&x_new, &x)?;
        let diff_norm: f64 = {
            let diff_sq = client.mul(&diff, &diff)?;
            let sum = client.sum(&diff_sq, &[], false)?;
            let sum_vec: Vec<f64> = sum.to_vec();
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

/// General matrix function f(A) using Schur-Parlett algorithm
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

    // Compute f(T) using shared algorithm
    let f_t = matrix_functions_core::funm_quasi_triangular_f64(&t_data, n, &f)?;

    // Reconstruct: f(A) = Z @ f(T) @ Z^T on GPU
    let f_t_tensor = Tensor::<CudaRuntime>::from_slice(&f_t, &[n, n], device);
    let z_tensor = Tensor::<CudaRuntime>::from_slice(&z_data, &[n, n], device);

    let temp = client.matmul(&z_tensor, &f_t_tensor)?;
    let z_t = z_tensor.transpose(0, 1)?;
    client.matmul(&temp, &z_t)
}
