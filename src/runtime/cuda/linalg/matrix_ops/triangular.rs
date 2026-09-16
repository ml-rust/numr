//! Triangular extraction and sign/log-determinant for CUDA — delegates to impl_generic

use super::super::super::CudaRuntime;
use super::super::super::client::CudaClient;
use crate::algorithm::linalg::SlogdetResult;
use crate::error::Result;
use crate::tensor::Tensor;

/// Upper triangular part of a matrix — delegates to impl_generic
pub fn triu_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    diagonal: i64,
) -> Result<Tensor<CudaRuntime>> {
    crate::ops::impl_generic::triu_impl(client, a, diagonal)
}

/// Lower triangular part of a matrix — delegates to impl_generic
pub fn tril_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    diagonal: i64,
) -> Result<Tensor<CudaRuntime>> {
    crate::ops::impl_generic::tril_impl(client, a, diagonal)
}

/// Sign and log-absolute-determinant — delegates to impl_generic
pub fn slogdet_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
) -> Result<SlogdetResult<CudaRuntime>> {
    crate::ops::impl_generic::slogdet_impl(client, a)
}
