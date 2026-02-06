//! WebGPU implementation of linear algebra algorithms.
//!
//! This module implements the [`LinearAlgebraAlgorithms`] and [`MatrixFunctionsAlgorithms`]
//! traits for WebGPU. All algorithms use native WGSL compute shaders - NO CPU FALLBACK.
//!
//! # Supported Data Types
//!
//! **F32 only.** WGSL (WebGPU Shading Language) does not natively support 64-bit
//! floating point operations. All functions in this module will return
//! [`Error::UnsupportedDType`] for non-F32 inputs.
//!
//! For F64 linear algebra, use the CPU or CUDA backends instead.
//!
//! # Performance Note
//!
//! Operations use native WGSL compute shaders running entirely on the GPU.

mod advanced_decompositions;
mod helpers;

mod decompositions;
mod eig_general;
mod eig_symmetric;
mod lstsq;
mod matrix_functions;
mod matrix_ops;
mod schur;
mod solvers;
mod statistics;
mod svd;
mod tensor_decompose;
mod triangular_solve;

#[cfg(test)]
mod tests;

use super::{WgpuClient, WgpuRuntime};
use crate::algorithm::linalg::{
    CholeskyDecomposition, ComplexSchurDecomposition, EigenDecomposition,
    GeneralEigenDecomposition, GeneralizedSchurDecomposition, LinearAlgebraAlgorithms,
    LuDecomposition, MatrixFunctionsAlgorithms, MatrixNormOrder, PolarDecomposition,
    QrDecomposition, SvdDecomposition,
};
use crate::error::Result;
use crate::tensor::Tensor;

// Implement LinearAlgebraAlgorithms for WgpuClient
impl LinearAlgebraAlgorithms<WgpuRuntime> for WgpuClient {
    fn lu_decompose(&self, a: &Tensor<WgpuRuntime>) -> Result<LuDecomposition<WgpuRuntime>> {
        decompositions::lu_decompose(self, a)
    }

    fn cholesky_decompose(
        &self,
        a: &Tensor<WgpuRuntime>,
    ) -> Result<CholeskyDecomposition<WgpuRuntime>> {
        decompositions::cholesky_decompose(self, a)
    }

    fn qr_decompose(&self, a: &Tensor<WgpuRuntime>) -> Result<QrDecomposition<WgpuRuntime>> {
        decompositions::qr_decompose_internal(self, a, false)
    }

    fn qr_decompose_thin(&self, a: &Tensor<WgpuRuntime>) -> Result<QrDecomposition<WgpuRuntime>> {
        decompositions::qr_decompose_internal(self, a, true)
    }

    fn solve(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        solvers::solve(self, a, b)
    }

    fn solve_triangular_lower(
        &self,
        l: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        unit_diagonal: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        solvers::solve_triangular_lower(self, l, b, unit_diagonal)
    }

    fn solve_triangular_upper(
        &self,
        u: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        solvers::solve_triangular_upper(self, u, b)
    }

    fn lstsq(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        solvers::lstsq(self, a, b)
    }

    fn inverse(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        matrix_ops::inverse(self, a)
    }

    fn det(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        matrix_ops::det(self, a)
    }

    fn trace(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        matrix_ops::trace(self, a)
    }

    fn diag(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        matrix_ops::diag(self, a)
    }

    fn diagflat(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        matrix_ops::diagflat(self, a)
    }

    fn kron(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        matrix_ops::kron(self, a, b)
    }

    fn khatri_rao(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        matrix_ops::khatri_rao(self, a, b)
    }

    fn matrix_rank(
        &self,
        a: &Tensor<WgpuRuntime>,
        tol: Option<f64>,
    ) -> Result<Tensor<WgpuRuntime>> {
        matrix_ops::matrix_rank(self, a, tol)
    }

    fn matrix_norm(
        &self,
        a: &Tensor<WgpuRuntime>,
        ord: MatrixNormOrder,
    ) -> Result<Tensor<WgpuRuntime>> {
        matrix_ops::matrix_norm(self, a, ord)
    }

    fn svd_decompose(&self, a: &Tensor<WgpuRuntime>) -> Result<SvdDecomposition<WgpuRuntime>> {
        svd::svd_decompose(self, a)
    }

    fn pinverse(&self, a: &Tensor<WgpuRuntime>, rcond: Option<f64>) -> Result<Tensor<WgpuRuntime>> {
        svd::pinverse(self, a, rcond)
    }

    fn cond(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        svd::cond(self, a)
    }

    fn cov(&self, a: &Tensor<WgpuRuntime>, ddof: Option<usize>) -> Result<Tensor<WgpuRuntime>> {
        statistics::cov(self, a, ddof)
    }

    fn corrcoef(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        statistics::corrcoef(self, a)
    }

    fn eig_decompose_symmetric(
        &self,
        a: &Tensor<WgpuRuntime>,
    ) -> Result<EigenDecomposition<WgpuRuntime>> {
        eig_symmetric::eig_decompose_symmetric(self, a)
    }

    fn schur_decompose(
        &self,
        a: &Tensor<WgpuRuntime>,
    ) -> Result<crate::algorithm::linalg::SchurDecomposition<WgpuRuntime>> {
        schur::schur_decompose(self, a)
    }

    fn eig_decompose(
        &self,
        a: &Tensor<WgpuRuntime>,
    ) -> Result<GeneralEigenDecomposition<WgpuRuntime>> {
        eig_general::eig_decompose(self, a)
    }

    fn rsf2csf(
        &self,
        schur: &crate::algorithm::linalg::SchurDecomposition<WgpuRuntime>,
    ) -> Result<ComplexSchurDecomposition<WgpuRuntime>> {
        advanced_decompositions::rsf2csf(self, schur)
    }

    fn qz_decompose(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<GeneralizedSchurDecomposition<WgpuRuntime>> {
        advanced_decompositions::qz_decompose(self, a, b)
    }

    fn polar_decompose(&self, a: &Tensor<WgpuRuntime>) -> Result<PolarDecomposition<WgpuRuntime>> {
        advanced_decompositions::polar_decompose(self, a)
    }

    fn triu(&self, a: &Tensor<WgpuRuntime>, diagonal: i64) -> Result<Tensor<WgpuRuntime>> {
        matrix_ops::triu(self, a, diagonal)
    }

    fn tril(&self, a: &Tensor<WgpuRuntime>, diagonal: i64) -> Result<Tensor<WgpuRuntime>> {
        matrix_ops::tril(self, a, diagonal)
    }

    fn slogdet(
        &self,
        a: &Tensor<WgpuRuntime>,
    ) -> Result<crate::algorithm::linalg::SlogdetResult<WgpuRuntime>> {
        matrix_ops::slogdet(self, a)
    }
}

// Implement MatrixFunctionsAlgorithms for WgpuClient
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
