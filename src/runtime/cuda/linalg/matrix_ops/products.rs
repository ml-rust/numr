//! Kronecker and Khatri-Rao products for CUDA

use super::super::super::CudaRuntime;
use super::super::super::client::CudaClient;
use super::super::super::kernels;
use crate::algorithm::linalg::validate_matrix_2d;
use crate::error::{Error, Result};
use crate::runtime::{AllocGuard, RuntimeClient};
use crate::tensor::Tensor;

/// Kronecker product: A ⊗ B
pub fn kron_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    crate::algorithm::linalg::validate_linalg_dtype(a.dtype())?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

    let (m_a, n_a) = validate_matrix_2d(a.shape())?;
    let (m_b, n_b) = validate_matrix_2d(b.shape())?;

    let dtype = a.dtype();
    let device = client.device();

    let m_out = m_a * m_b;
    let n_out = n_a * n_b;
    let out_size = m_out * n_out * dtype.size_in_bytes();
    let out_guard = AllocGuard::new(client.allocator(), out_size)?;
    let out_ptr = out_guard.ptr();

    unsafe {
        kernels::launch_kron(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            a.ptr(),
            b.ptr(),
            out_ptr,
            m_a,
            n_a,
            m_b,
            n_b,
        )?;
    }

    client.synchronize();

    let out =
        unsafe { CudaClient::tensor_from_raw(out_guard.release(), &[m_out, n_out], dtype, device) };

    Ok(out)
}

/// Khatri-Rao product (column-wise Kronecker): A ⊙ B
///
/// For A of shape [m, k] and B of shape [n, k],
/// produces output of shape [m * n, k].
///
/// (A ⊙ B)[i*n + j, c] = A[i, c] * B[j, c]
pub fn khatri_rao_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    crate::algorithm::linalg::validate_linalg_dtype(a.dtype())?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

    let (m, k_a) = validate_matrix_2d(a.shape())?;
    let (n, k_b) = validate_matrix_2d(b.shape())?;

    if k_a != k_b {
        return Err(Error::Internal(format!(
            "khatri_rao: column count mismatch. A has shape [{}, {}], B has shape [{}, {}]. \
             Matrices must have the same number of columns.",
            m, k_a, n, k_b
        )));
    }

    let k = k_a;
    let dtype = a.dtype();
    let device = client.device();

    let m_out = m * n;
    let out_size = m_out * k * dtype.size_in_bytes();
    let out_guard = AllocGuard::new(client.allocator(), out_size)?;
    let out_ptr = out_guard.ptr();

    unsafe {
        kernels::launch_khatri_rao(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            a.ptr(),
            b.ptr(),
            out_ptr,
            m,
            n,
            k,
        )?;
    }

    client.synchronize();

    let out =
        unsafe { CudaClient::tensor_from_raw(out_guard.release(), &[m_out, k], dtype, device) };

    Ok(out)
}
