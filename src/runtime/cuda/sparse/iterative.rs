//! CUDA implementation of iterative solvers.
//!
//! This module provides CUDA implementations of iterative solvers
//! using the generic algorithms in `algorithm::iterative_impl`.

use super::{CudaClient, CudaRuntime};
use crate::algorithm::iterative::{
    BiCgStabOptions, BiCgStabResult, GmresOptions, GmresResult, IterativeSolvers,
};
use crate::algorithm::iterative::{bicgstab_impl, gmres_impl};
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
}
