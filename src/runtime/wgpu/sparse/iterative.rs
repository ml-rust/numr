//! WebGPU implementation of iterative solvers.
//!
//! This module provides WebGPU implementations of iterative solvers
//! using the generic algorithms in `algorithm::iterative_impl`.

use super::super::{WgpuClient, WgpuRuntime};
use crate::algorithm::iterative::{
    BiCgStabOptions, BiCgStabResult, GmresOptions, GmresResult, IterativeSolvers,
};
use crate::algorithm::iterative_impl::{bicgstab_impl, gmres_impl};
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
}
