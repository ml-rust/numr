//! `MatrixFunctionsAlgorithms` impl for `WgpuClient`.

use super::super::{WgpuClient, WgpuRuntime};
use super::matrix_functions;
use crate::algorithm::linalg::MatrixFunctionsAlgorithms;
use crate::error::Result;
use crate::tensor::Tensor;

impl MatrixFunctionsAlgorithms<WgpuRuntime> for WgpuClient {
    fn expm(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        matrix_functions::expm(self, a)
    }

    fn logm(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        matrix_functions::logm(self, a)
    }

    fn sqrtm(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        matrix_functions::sqrtm(self, a)
    }

    fn signm(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        matrix_functions::signm(self, a)
    }

    fn fractional_matrix_power(
        &self,
        a: &Tensor<WgpuRuntime>,
        p: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        matrix_functions::fractional_matrix_power(self, a, p)
    }

    fn funm<F>(&self, a: &Tensor<WgpuRuntime>, f: F) -> Result<Tensor<WgpuRuntime>>
    where
        F: Fn(f64) -> f64 + Send + Sync,
    {
        matrix_functions::funm(self, a, f)
    }
}
