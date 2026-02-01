//! Linear algebra operations for CUDA runtime
use crate::error::Result;
use crate::ops::LinalgOps;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl LinalgOps<CudaRuntime> for CudaClient {
    fn solve(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::solve(self, a, b)
    }

    fn lstsq(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::lstsq(self, a, b)
    }

    fn pinverse(&self, a: &Tensor<CudaRuntime>, rcond: Option<f64>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::pinverse(self, a, rcond)
    }

    fn matrix_norm(
        &self,
        a: &Tensor<CudaRuntime>,
        ord: crate::algorithm::linalg::MatrixNormOrder,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::matrix_norm(self, a, ord)
    }

    fn inverse(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::inverse(self, a)
    }

    fn det(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::det(self, a)
    }

    fn trace(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::trace(self, a)
    }

    fn diag(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::diag(self, a)
    }

    fn diagflat(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::diagflat(self, a)
    }

    fn matrix_rank(
        &self,
        a: &Tensor<CudaRuntime>,
        tol: Option<f64>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::linalg::LinearAlgebraAlgorithms;
        LinearAlgebraAlgorithms::matrix_rank(self, a, tol)
    }
}
