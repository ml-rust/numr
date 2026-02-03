//! Sparse tensor operations for WebGPU runtime.
//!
//! This module provides GPU-accelerated sparse linear algebra operations
//! using level scheduling for parallel execution.

mod common;
mod ic0;
mod ilu0;
mod triangular_solve;

use super::{WgpuClient, WgpuRuntime};
use crate::algorithm::sparse_linalg::{
    IcDecomposition, IcOptions, IluDecomposition, IluOptions, SparseLinAlgAlgorithms,
};
use crate::error::Result;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

impl SparseLinAlgAlgorithms<WgpuRuntime> for WgpuClient {
    fn ilu0(
        &self,
        a: &CsrData<WgpuRuntime>,
        options: IluOptions,
    ) -> Result<IluDecomposition<WgpuRuntime>> {
        ilu0::ilu0_wgpu(self, a, options)
    }

    fn ic0(
        &self,
        a: &CsrData<WgpuRuntime>,
        options: IcOptions,
    ) -> Result<IcDecomposition<WgpuRuntime>> {
        ic0::ic0_wgpu(self, a, options)
    }

    fn sparse_solve_triangular(
        &self,
        l_or_u: &CsrData<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        triangular_solve::sparse_solve_triangular_wgpu(self, l_or_u, b, lower, unit_diagonal)
    }
}
