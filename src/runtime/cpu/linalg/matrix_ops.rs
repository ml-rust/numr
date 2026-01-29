//! Basic matrix operations (inverse, det, trace, diag, rank, norms)

use super::super::jacobi::LinalgElement;
use super::super::{CpuClient, CpuRuntime};
use super::decompositions::{lu_decompose_impl, qr_decompose_impl};
use super::solvers::solve_impl;
use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Matrix inverse via LU decomposition
pub fn inverse_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Create identity matrix
    let mut identity: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        identity[i * n + i] = T::one();
    }
    let identity_tensor = Tensor::<CpuRuntime>::from_slice(&identity, &[n, n], device);

    // Solve A @ X = I
    solve_impl::<T>(client, a, &identity_tensor, n)
}

/// Determinant via LU decomposition
pub fn det_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Handle special case n=0
    if n == 0 {
        return Ok(Tensor::<CpuRuntime>::from_slice(&[T::one()], &[], device));
    }

    // Compute LU decomposition
    let lu_decomp = lu_decompose_impl::<T>(client, a, n, n)?;
    let lu_data: Vec<T> = lu_decomp.lu.to_vec();

    // det = (-1)^num_swaps * product(U[i,i])
    let mut det = if lu_decomp.num_swaps % 2 == 0 {
        T::one()
    } else {
        T::one().neg_val()
    };

    for i in 0..n {
        det = det * lu_data[i * n + i];
    }

    Ok(Tensor::<CpuRuntime>::from_slice(&[det], &[], device))
}

/// Trace: sum of diagonal elements
pub fn trace_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();

    let k = m.min(n);
    let mut trace = T::zero();
    for i in 0..k {
        trace = trace + a_data[i * n + i];
    }

    Ok(Tensor::<CpuRuntime>::from_slice(&[trace], &[], device))
}

/// Extract diagonal
pub fn diag_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();

    let k = m.min(n);
    let mut diag: Vec<T> = vec![T::zero(); k];
    for i in 0..k {
        diag[i] = a_data[i * n + i];
    }

    Ok(Tensor::<CpuRuntime>::from_slice(&diag, &[k], device))
}

/// Create diagonal matrix from 1D tensor
pub fn diagflat_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let n = a.shape()[0];
    let a_data: Vec<T> = a.to_vec();

    let mut mat: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        mat[i * n + i] = a_data[i];
    }

    Ok(Tensor::<CpuRuntime>::from_slice(&mat, &[n, n], device))
}

/// Matrix rank via singular value thresholding
/// Uses QR-based approach since SVD is not yet implemented
pub fn matrix_rank_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
    tol: Option<f64>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Use QR decomposition to estimate rank (R diagonal gives singular value bounds)
    let qr = qr_decompose_impl::<T>(client, a, m, n, true)?;
    let r_data: Vec<T> = qr.r.to_vec();

    let k = m.min(n);

    // Find max diagonal element of R (upper bound on max singular value)
    let mut max_diag = T::zero();
    for i in 0..k {
        let val = r_data[i * n + i].abs_val();
        if val.to_f64() > max_diag.to_f64() {
            max_diag = val;
        }
    }

    // Compute tolerance
    let eps = if a.dtype() == DType::F32 {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    };
    let tolerance = tol.unwrap_or_else(|| m.max(n) as f64 * eps * max_diag.to_f64());

    // Count diagonal elements above tolerance
    let mut rank = 0i64;
    for i in 0..k {
        if r_data[i * n + i].abs_val().to_f64() > tolerance {
            rank += 1;
        }
    }

    Ok(Tensor::<CpuRuntime>::from_slice(&[rank], &[], device))
}

/// Frobenius norm: ||A||_F = sqrt(sum_{i,j} |A[i,j]|^2)
pub fn frobenius_norm_impl<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();

    // Sum of squares of all elements
    let mut sum_sq = T::zero();
    for &val in &a_data {
        sum_sq = sum_sq + val * val;
    }

    let norm = sum_sq.sqrt_val();
    Ok(Tensor::<CpuRuntime>::from_slice(&[norm], &[], device))
}
