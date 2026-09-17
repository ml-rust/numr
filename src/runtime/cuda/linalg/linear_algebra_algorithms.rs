//! `LinearAlgebraAlgorithms` impl for `CudaClient`.

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::{
    advanced_decompositions, banded, decompositions, eig_general, eig_symmetric, matrix_ops, schur,
    solvers, statistics, svd,
};
use crate::algorithm::linalg::{
    CholeskyDecomposition, ComplexSchurDecomposition, EigenDecomposition,
    GeneralEigenDecomposition, GeneralizedSchurDecomposition, LinearAlgebraAlgorithms,
    LuDecomposition, MatrixNormOrder, PolarDecomposition, QrDecomposition, SchurDecomposition,
    SvdDecomposition,
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

    fn solve_banded(
        &self,
        ab: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        kl: usize,
        ku: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        banded::solve_banded_impl(self, ab, b, kl, ku)
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

    fn kron(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        matrix_ops::kron_impl(self, a, b)
    }

    fn triu(&self, a: &Tensor<CudaRuntime>, diagonal: i64) -> Result<Tensor<CudaRuntime>> {
        matrix_ops::triu_impl(self, a, diagonal)
    }

    fn tril(&self, a: &Tensor<CudaRuntime>, diagonal: i64) -> Result<Tensor<CudaRuntime>> {
        matrix_ops::tril_impl(self, a, diagonal)
    }

    fn slogdet(
        &self,
        a: &Tensor<CudaRuntime>,
    ) -> Result<crate::algorithm::linalg::SlogdetResult<CudaRuntime>> {
        matrix_ops::slogdet_impl(self, a)
    }

    fn khatri_rao(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        matrix_ops::khatri_rao_impl(self, a, b)
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

    fn rsf2csf(
        &self,
        schur: &SchurDecomposition<CudaRuntime>,
    ) -> Result<ComplexSchurDecomposition<CudaRuntime>> {
        advanced_decompositions::rsf2csf_impl(self, schur)
    }

    fn qz_decompose(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<GeneralizedSchurDecomposition<CudaRuntime>> {
        advanced_decompositions::qz_decompose_impl(self, a, b)
    }

    fn polar_decompose(&self, a: &Tensor<CudaRuntime>) -> Result<PolarDecomposition<CudaRuntime>> {
        advanced_decompositions::polar_decompose_impl(self, a)
    }
}
