//! WebGPU implementation of iterative solvers.
//!
//! This module provides WebGPU implementations of iterative solvers
//! using the generic algorithms in `algorithm::iterative::impl_generic`.

use super::super::{WgpuClient, WgpuRuntime};
use crate::algorithm::iterative::{
    BiCgStabOptions, BiCgStabResult, CgOptions, CgResult, CgsOptions, CgsResult, GmresOptions,
    GmresResult, IterativeSolvers, JacobiOptions, JacobiResult, LgmresOptions, LgmresResult,
    MinresOptions, MinresResult, QmrOptions, QmrResult, SorOptions, SorResult,
    SparseEigComplexResult, SparseEigOptions, SparseEigResult, SparseSvdResult, SvdsOptions,
};
use crate::algorithm::iterative::{
    arnoldi_eig_impl, bicgstab_impl, cg_impl, cgs_impl, gmres_impl, jacobi_impl, lanczos_eig_impl,
    lgmres_impl, minres_impl, qmr_impl, sor_impl, svds_impl,
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

    fn lgmres(
        &self,
        a: &CsrData<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        x0: Option<&Tensor<WgpuRuntime>>,
        options: LgmresOptions,
    ) -> Result<LgmresResult<WgpuRuntime>> {
        lgmres_impl(self, a, b, x0, options)
    }

    fn qmr(
        &self,
        a: &CsrData<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        x0: Option<&Tensor<WgpuRuntime>>,
        options: QmrOptions,
    ) -> Result<QmrResult<WgpuRuntime>> {
        qmr_impl(self, a, b, x0, options)
    }

    fn jacobi(
        &self,
        a: &CsrData<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        x0: Option<&Tensor<WgpuRuntime>>,
        options: JacobiOptions,
    ) -> Result<JacobiResult<WgpuRuntime>> {
        jacobi_impl(self, a, b, x0, options)
    }

    fn sor(
        &self,
        a: &CsrData<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        x0: Option<&Tensor<WgpuRuntime>>,
        options: SorOptions,
    ) -> Result<SorResult<WgpuRuntime>> {
        sor_impl(self, a, b, x0, options)
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

    fn svds(
        &self,
        a: &CsrData<WgpuRuntime>,
        k: usize,
        options: SvdsOptions,
    ) -> Result<SparseSvdResult<WgpuRuntime>> {
        svds_impl(self, a, k, options)
    }
}
