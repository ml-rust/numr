//! CUDA implementation of sparse linear algebra algorithms.
//!
//! This module provides GPU-accelerated sparse linear algebra using level scheduling
//! for parallel execution of inherently sequential algorithms.
//!
//! # Level Scheduling
//!
//! ILU(0), IC(0), and sparse triangular solve have row-to-row dependencies.
//! Level scheduling analyzes the sparsity pattern to find independent rows:
//! - Rows at the same level can execute in parallel
//! - Levels execute sequentially
//!
//! This enables GPU parallelism while maintaining correctness.

mod common;
mod ic0;
mod ilu0;
mod iluk;
mod triangular_solve;

use super::{CudaClient, CudaRuntime};
use crate::algorithm::sparse_linalg::{
    IcDecomposition, IcOptions, IluDecomposition, IluFillLevel, IluOptions, IlukDecomposition,
    IlukOptions, IlukSymbolic, SparseLinAlgAlgorithms, SymbolicIlu0,
};
use crate::error::Result;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

impl SparseLinAlgAlgorithms<CudaRuntime> for CudaClient {
    fn ilu0(
        &self,
        a: &CsrData<CudaRuntime>,
        options: IluOptions,
    ) -> Result<IluDecomposition<CudaRuntime>> {
        ilu0::ilu0_cuda(self, a, options)
    }

    fn ic0(
        &self,
        a: &CsrData<CudaRuntime>,
        options: IcOptions,
    ) -> Result<IcDecomposition<CudaRuntime>> {
        ic0::ic0_cuda(self, a, options)
    }

    fn sparse_solve_triangular(
        &self,
        l_or_u: &CsrData<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        triangular_solve::sparse_solve_triangular_cuda(self, l_or_u, b, lower, unit_diagonal)
    }

    fn iluk_symbolic(&self, a: &CsrData<CudaRuntime>, level: IluFillLevel) -> Result<IlukSymbolic> {
        iluk::iluk_symbolic_cuda(self, a, level)
    }

    fn iluk_numeric(
        &self,
        a: &CsrData<CudaRuntime>,
        symbolic: &IlukSymbolic,
        opts: &IlukOptions,
    ) -> Result<IlukDecomposition<CudaRuntime>> {
        iluk::iluk_numeric_cuda(self, a, symbolic, opts)
    }

    fn iluk(
        &self,
        a: &CsrData<CudaRuntime>,
        opts: IlukOptions,
    ) -> Result<IlukDecomposition<CudaRuntime>> {
        iluk::iluk_cuda(self, a, opts)
    }

    fn ilu0_symbolic(&self, pattern: &CsrData<CudaRuntime>) -> Result<SymbolicIlu0> {
        ilu0::ilu0_symbolic_cuda(self, pattern)
    }

    fn ilu0_numeric(
        &self,
        a: &CsrData<CudaRuntime>,
        symbolic: &SymbolicIlu0,
        options: IluOptions,
    ) -> Result<IluDecomposition<CudaRuntime>> {
        ilu0::ilu0_numeric_cuda(self, a, symbolic, options)
    }
}
