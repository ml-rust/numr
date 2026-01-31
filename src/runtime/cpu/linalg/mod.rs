//! CPU implementation of linear algebra algorithms
//!
//! This module implements the [`LinearAlgebraAlgorithms`] trait for CPU.
//! All algorithms follow the exact specification in the trait documentation
//! to ensure backend parity with CUDA/WebGPU implementations.

mod advanced_decompositions;
mod decompositions;
mod eig_general;
mod eig_symmetric;
mod matrix_functions;
mod matrix_ops;
mod schur;
mod solvers;
mod statistics;
mod svd;

#[cfg(test)]
mod tests;

use super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::{
    CholeskyDecomposition, ComplexSchurDecomposition, EigenDecomposition,
    GeneralEigenDecomposition, GeneralizedSchurDecomposition, LinearAlgebraAlgorithms,
    LuDecomposition, MatrixFunctionsAlgorithms, MatrixNormOrder, PolarDecomposition,
    QrDecomposition, SchurDecomposition, SvdDecomposition,
};
use crate::error::Result;
use crate::tensor::Tensor;

impl LinearAlgebraAlgorithms<CpuRuntime> for CpuClient {
    fn lu_decompose(&self, a: &Tensor<CpuRuntime>) -> Result<LuDecomposition<CpuRuntime>> {
        decompositions::lu_decompose_impl(self, a)
    }

    fn cholesky_decompose(
        &self,
        a: &Tensor<CpuRuntime>,
    ) -> Result<CholeskyDecomposition<CpuRuntime>> {
        decompositions::cholesky_decompose_impl(self, a)
    }

    fn qr_decompose(&self, a: &Tensor<CpuRuntime>) -> Result<QrDecomposition<CpuRuntime>> {
        decompositions::qr_decompose_impl(self, a, false)
    }

    fn qr_decompose_thin(&self, a: &Tensor<CpuRuntime>) -> Result<QrDecomposition<CpuRuntime>> {
        decompositions::qr_decompose_impl(self, a, true)
    }

    fn solve(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        solvers::solve_impl(self, a, b)
    }

    fn solve_triangular_lower(
        &self,
        l: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        unit_diagonal: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        solvers::solve_triangular_lower_impl(self, l, b, unit_diagonal)
    }

    fn solve_triangular_upper(
        &self,
        u: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        solvers::solve_triangular_upper_impl(self, u, b)
    }

    fn lstsq(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        solvers::lstsq_impl(self, a, b)
    }

    fn inverse(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        matrix_ops::inverse_impl(self, a)
    }

    fn det(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        matrix_ops::det_impl(self, a)
    }

    fn trace(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        matrix_ops::trace_impl(self, a)
    }

    fn diag(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        matrix_ops::diag_impl(self, a)
    }

    fn diagflat(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        matrix_ops::diagflat_impl(self, a)
    }

    fn kron(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        matrix_ops::kron_impl(self, a, b)
    }

    fn matrix_rank(&self, a: &Tensor<CpuRuntime>, tol: Option<f64>) -> Result<Tensor<CpuRuntime>> {
        matrix_ops::matrix_rank_impl(self, a, tol)
    }

    fn matrix_norm(
        &self,
        a: &Tensor<CpuRuntime>,
        ord: MatrixNormOrder,
    ) -> Result<Tensor<CpuRuntime>> {
        matrix_ops::matrix_norm_impl(self, a, ord)
    }

    fn svd_decompose(&self, a: &Tensor<CpuRuntime>) -> Result<SvdDecomposition<CpuRuntime>> {
        svd::svd_decompose_impl(self, a)
    }

    fn eig_decompose_symmetric(
        &self,
        a: &Tensor<CpuRuntime>,
    ) -> Result<EigenDecomposition<CpuRuntime>> {
        eig_symmetric::eig_decompose_symmetric_impl(self, a)
    }

    fn pinverse(&self, a: &Tensor<CpuRuntime>, rcond: Option<f64>) -> Result<Tensor<CpuRuntime>> {
        statistics::pinverse_impl(self, a, rcond)
    }

    fn cond(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        statistics::cond_impl(self, a)
    }

    fn cov(&self, a: &Tensor<CpuRuntime>, ddof: Option<usize>) -> Result<Tensor<CpuRuntime>> {
        statistics::cov_impl(self, a, ddof)
    }

    fn corrcoef(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        statistics::corrcoef_impl(self, a)
    }

    fn schur_decompose(&self, a: &Tensor<CpuRuntime>) -> Result<SchurDecomposition<CpuRuntime>> {
        schur::schur_decompose_impl(self, a)
    }

    fn eig_decompose(
        &self,
        a: &Tensor<CpuRuntime>,
    ) -> Result<GeneralEigenDecomposition<CpuRuntime>> {
        eig_general::eig_decompose_impl(self, a)
    }

    fn rsf2csf(
        &self,
        schur: &SchurDecomposition<CpuRuntime>,
    ) -> Result<ComplexSchurDecomposition<CpuRuntime>> {
        advanced_decompositions::rsf2csf_impl(self, schur)
    }

    fn qz_decompose(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<GeneralizedSchurDecomposition<CpuRuntime>> {
        advanced_decompositions::qz_decompose_impl(self, a, b)
    }

    fn polar_decompose(&self, a: &Tensor<CpuRuntime>) -> Result<PolarDecomposition<CpuRuntime>> {
        advanced_decompositions::polar_decompose_impl(self, a)
    }
}

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
