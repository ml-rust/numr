//! CUDA implementation of linear algebra algorithms
//!
//! This module implements the [`LinearAlgebraAlgorithms`] trait for CUDA.
//! All algorithms follow the exact specification in the trait documentation
//! to ensure backend parity with CPU/WebGPU implementations.
//!
//! Native CUDA kernels are used - NO cuSOLVER dependency.

mod decompositions;
mod eig_general;
mod eig_symmetric;
pub(crate) mod helpers;
mod matrix_functions;
mod matrix_ops;
mod schur;
mod solvers;
mod statistics;
mod svd;

#[cfg(test)]
mod tests;

use super::CudaRuntime;
use super::client::CudaClient;
use crate::algorithm::linalg::{
    CholeskyDecomposition, EigenDecomposition, GeneralEigenDecomposition, LinearAlgebraAlgorithms,
    LuDecomposition, MatrixFunctionsAlgorithms, MatrixNormOrder, QrDecomposition,
    SchurDecomposition, SvdDecomposition,
};
use crate::error::Result;
use crate::tensor::Tensor;

impl LinearAlgebraAlgorithms<CudaRuntime> for CudaClient {
    fn lu_decompose(&self, a: &Tensor<CudaRuntime>) -> Result<LuDecomposition<CudaRuntime>> {
        decompositions::lu_decompose_impl(self, a)
    }

    fn cholesky_decompose(
        &self,
        a: &Tensor<CudaRuntime>,
    ) -> Result<CholeskyDecomposition<CudaRuntime>> {
        decompositions::cholesky_decompose_impl(self, a)
    }

    fn qr_decompose(&self, a: &Tensor<CudaRuntime>) -> Result<QrDecomposition<CudaRuntime>> {
        decompositions::qr_decompose_internal(self, a, false)
    }

    fn qr_decompose_thin(&self, a: &Tensor<CudaRuntime>) -> Result<QrDecomposition<CudaRuntime>> {
        decompositions::qr_decompose_internal(self, a, true)
    }

    fn solve(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        solvers::solve_impl(self, a, b)
    }

    fn solve_triangular_lower(
        &self,
        l: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        unit_diagonal: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        solvers::solve_triangular_lower_impl(self, l, b, unit_diagonal)
    }

    fn solve_triangular_upper(
        &self,
        u: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        solvers::solve_triangular_upper_impl(self, u, b)
    }

    fn lstsq(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        solvers::lstsq_impl(self, a, b)
    }

    fn inverse(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        matrix_ops::inverse_impl(self, a)
    }

    fn det(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        matrix_ops::det_impl(self, a)
    }

    fn trace(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        matrix_ops::trace_impl(self, a)
    }

    fn diag(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        matrix_ops::diag_impl(self, a)
    }

    fn diagflat(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        matrix_ops::diagflat_impl(self, a)
    }

    fn matrix_rank(
        &self,
        a: &Tensor<CudaRuntime>,
        tol: Option<f64>,
    ) -> Result<Tensor<CudaRuntime>> {
        matrix_ops::matrix_rank_impl(self, a, tol)
    }

    fn matrix_norm(
        &self,
        a: &Tensor<CudaRuntime>,
        ord: MatrixNormOrder,
    ) -> Result<Tensor<CudaRuntime>> {
        matrix_ops::matrix_norm_impl(self, a, ord)
    }

    fn svd_decompose(&self, a: &Tensor<CudaRuntime>) -> Result<SvdDecomposition<CudaRuntime>> {
        svd::svd_decompose_impl(self, a)
    }

    fn pinverse(&self, a: &Tensor<CudaRuntime>, rcond: Option<f64>) -> Result<Tensor<CudaRuntime>> {
        statistics::pinverse_impl(self, a, rcond)
    }

    fn cond(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        statistics::cond_impl(self, a)
    }

    fn cov(&self, a: &Tensor<CudaRuntime>, ddof: Option<usize>) -> Result<Tensor<CudaRuntime>> {
        statistics::cov_impl(self, a, ddof)
    }

    fn corrcoef(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        statistics::corrcoef_impl(self, a)
    }

    fn eig_decompose_symmetric(
        &self,
        a: &Tensor<CudaRuntime>,
    ) -> Result<EigenDecomposition<CudaRuntime>> {
        eig_symmetric::eig_decompose_symmetric_impl(self, a)
    }

    fn schur_decompose(&self, a: &Tensor<CudaRuntime>) -> Result<SchurDecomposition<CudaRuntime>> {
        schur::schur_decompose_impl(self, a)
    }

    fn eig_decompose(
        &self,
        a: &Tensor<CudaRuntime>,
    ) -> Result<GeneralEigenDecomposition<CudaRuntime>> {
        eig_general::eig_decompose_impl(self, a)
    }
}

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
