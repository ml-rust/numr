//! `MatrixFunctionsAlgorithms` impl for `CudaClient`.

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::matrix_functions;
use crate::algorithm::linalg::MatrixFunctionsAlgorithms;
use crate::error::Result;
use crate::tensor::Tensor;

impl MatrixFunctionsAlgorithms<CudaRuntime> for CudaClient {
    fn expm(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        matrix_functions::expm_impl(self, a)
    }

    fn logm(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        matrix_functions::logm_impl(self, a)
    }

    fn sqrtm(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        matrix_functions::sqrtm_impl(self, a)
    }

    fn signm(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        matrix_functions::signm_impl(self, a)
    }

    fn fractional_matrix_power(
        &self,
        a: &Tensor<CudaRuntime>,
        p: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        matrix_functions::fractional_matrix_power_impl(self, a, p)
    }

    fn funm<F>(&self, a: &Tensor<CudaRuntime>, f: F) -> Result<Tensor<CudaRuntime>>
    where
        F: Fn(f64) -> f64 + Send + Sync,
    {
        matrix_functions::funm_impl(self, a, f)
    }
}
