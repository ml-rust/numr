//! CPU implementation of linear algebra operations.

use crate::error::Result;
use crate::ops::LinalgOps;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

/// LinalgOps implementation for CPU runtime.
impl LinalgOps<CpuRuntime> for CpuClient {
    fn solve(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::solve(self, a, b)
    }

    fn lstsq(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::lstsq(self, a, b)
    }

    fn pinverse(&self, a: &Tensor<CpuRuntime>, rcond: Option<f64>) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::pinverse(self, a, rcond)
    }

    fn matrix_norm(
        &self,
        a: &Tensor<CpuRuntime>,
        ord: crate::algorithm::linalg::MatrixNormOrder,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::matrix_norm(self, a, ord)
    }

    fn inverse(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::inverse(self, a)
    }

    fn det(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::det(self, a)
    }

    fn trace(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::trace(self, a)
    }

    fn diag(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::diag(self, a)
    }

    fn diagflat(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::diagflat(self, a)
    }

    fn matrix_rank(&self, a: &Tensor<CpuRuntime>, tol: Option<f64>) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::matrix_rank(self, a, tol)
    }

    fn kron(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::kron(self, a, b)
    }

    fn solve_banded(
        &self,
        ab: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        kl: usize,
        ku: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::solve_banded(self, ab, b, kl, ku)
    }

    fn khatri_rao(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::khatri_rao(self, a, b)
    }

    fn triu(&self, a: &Tensor<CpuRuntime>, diagonal: i64) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::triu(self, a, diagonal)
    }

    fn tril(&self, a: &Tensor<CpuRuntime>, diagonal: i64) -> Result<Tensor<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::tril(self, a, diagonal)
    }

    fn slogdet(
        &self,
        a: &Tensor<CpuRuntime>,
    ) -> Result<crate::algorithm::linalg::SlogdetResult<CpuRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::slogdet(self, a)
    }
}
