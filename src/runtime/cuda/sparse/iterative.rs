//! CUDA implementation of iterative solvers.
//!
//! This module provides CUDA implementations of iterative solvers
//! using the generic algorithms in `algorithm::iterative::impl_generic`.

use super::{CudaClient, CudaRuntime};
use crate::algorithm::iterative::{
    BiCgStabOptions, BiCgStabResult, CgOptions, CgResult, CgsOptions, CgsResult, GmresOptions,
    GmresResult, IterativeSolvers, MinresOptions, MinresResult, SparseEigComplexResult,
    SparseEigOptions, SparseEigResult,
};
use crate::algorithm::iterative::{
    arnoldi_eig_impl, bicgstab_impl, cg_impl, cgs_impl, gmres_impl, lanczos_eig_impl, minres_impl,
};
use crate::error::Result;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

impl IterativeSolvers<CudaRuntime> for CudaClient {
    fn gmres(
        &self,
        a: &CsrData<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        x0: Option<&Tensor<CudaRuntime>>,
        options: GmresOptions,
    ) -> Result<GmresResult<CudaRuntime>> {
        gmres_impl(self, a, b, x0, options)
    }

    fn bicgstab(
        &self,
        a: &CsrData<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        x0: Option<&Tensor<CudaRuntime>>,
        options: BiCgStabOptions,
    ) -> Result<BiCgStabResult<CudaRuntime>> {
        bicgstab_impl(self, a, b, x0, options)
    }

    fn cg(
        &self,
        a: &CsrData<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        x0: Option<&Tensor<CudaRuntime>>,
        options: CgOptions,
    ) -> Result<CgResult<CudaRuntime>> {
        cg_impl(self, a, b, x0, options)
    }

    fn minres(
        &self,
        a: &CsrData<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        x0: Option<&Tensor<CudaRuntime>>,
        options: MinresOptions,
    ) -> Result<MinresResult<CudaRuntime>> {
        minres_impl(self, a, b, x0, options)
    }

    fn cgs(
        &self,
        a: &CsrData<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        x0: Option<&Tensor<CudaRuntime>>,
        options: CgsOptions,
    ) -> Result<CgsResult<CudaRuntime>> {
        cgs_impl(self, a, b, x0, options)
    }

    fn sparse_eig_symmetric(
        &self,
        a: &CsrData<CudaRuntime>,
        k: usize,
        options: SparseEigOptions,
    ) -> Result<SparseEigResult<CudaRuntime>> {
        lanczos_eig_impl(self, a, k, options)
    }

    fn sparse_eig(
        &self,
        a: &CsrData<CudaRuntime>,
        k: usize,
        options: SparseEigOptions,
    ) -> Result<SparseEigComplexResult<CudaRuntime>> {
        arnoldi_eig_impl(self, a, k, options)
    }
}
