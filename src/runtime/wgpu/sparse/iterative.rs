//! WebGPU implementation of iterative solvers.
//!
//! This module provides WebGPU implementations of iterative solvers
//! using the generic algorithms in `algorithm::iterative::impl_generic`.

use super::super::{WgpuClient, WgpuRuntime};
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

impl IterativeSolvers<WgpuRuntime> for WgpuClient {
    fn gmres(
        &self,
        a: &CsrData<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        x0: Option<&Tensor<WgpuRuntime>>,
        options: GmresOptions,
    ) -> Result<GmresResult<WgpuRuntime>> {
        gmres_impl(self, a, b, x0, options)
    }

    fn bicgstab(
        &self,
        a: &CsrData<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        x0: Option<&Tensor<WgpuRuntime>>,
        options: BiCgStabOptions,
    ) -> Result<BiCgStabResult<WgpuRuntime>> {
        bicgstab_impl(self, a, b, x0, options)
    }

    fn cg(
        &self,
        a: &CsrData<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        x0: Option<&Tensor<WgpuRuntime>>,
        options: CgOptions,
    ) -> Result<CgResult<WgpuRuntime>> {
        cg_impl(self, a, b, x0, options)
    }

    fn minres(
        &self,
        a: &CsrData<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        x0: Option<&Tensor<WgpuRuntime>>,
        options: MinresOptions,
    ) -> Result<MinresResult<WgpuRuntime>> {
        minres_impl(self, a, b, x0, options)
    }

    fn cgs(
        &self,
        a: &CsrData<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        x0: Option<&Tensor<WgpuRuntime>>,
        options: CgsOptions,
    ) -> Result<CgsResult<WgpuRuntime>> {
        cgs_impl(self, a, b, x0, options)
    }

    fn sparse_eig_symmetric(
        &self,
        a: &CsrData<WgpuRuntime>,
        k: usize,
        options: SparseEigOptions,
    ) -> Result<SparseEigResult<WgpuRuntime>> {
        lanczos_eig_impl(self, a, k, options)
    }

    fn sparse_eig(
        &self,
        a: &CsrData<WgpuRuntime>,
        k: usize,
        options: SparseEigOptions,
    ) -> Result<SparseEigComplexResult<WgpuRuntime>> {
        arnoldi_eig_impl(self, a, k, options)
    }
}
