use super::super::{CpuClient, CpuRuntime};
use super::matrix_functions;
use crate::algorithm::linalg::MatrixFunctionsAlgorithms;
use crate::error::Result;
use crate::tensor::Tensor;

impl MatrixFunctionsAlgorithms<CpuRuntime> for CpuClient {
    fn expm(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        matrix_functions::expm_impl(self, a)
    }

    fn logm(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        matrix_functions::logm_impl(self, a)
    }

    fn sqrtm(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        matrix_functions::sqrtm_impl(self, a)
    }

    fn signm(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        matrix_functions::signm_impl(self, a)
    }

    fn fractional_matrix_power(
        &self,
        a: &Tensor<CpuRuntime>,
        p: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        matrix_functions::fractional_matrix_power_impl(self, a, p)
    }

    fn funm<F>(&self, a: &Tensor<CpuRuntime>, f: F) -> Result<Tensor<CpuRuntime>>
    where
        F: Fn(f64) -> f64 + Send + Sync,
    {
        matrix_functions::funm_impl(self, a, f)
    }
}
