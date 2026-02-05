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
mod triangular_solve;

use super::{CudaClient, CudaRuntime};
use crate::algorithm::sparse_linalg::{
    IcDecomposition, IcOptions, IluDecomposition, IluFillLevel, IluOptions, IlukDecomposition,
    IlukOptions, IlukSymbolic, SparseLinAlgAlgorithms, SymbolicIlu0,
};
use crate::error::Error;
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

    fn iluk_symbolic(
        &self,
        _a: &CsrData<CudaRuntime>,
        _level: IluFillLevel,
    ) -> Result<IlukSymbolic> {
        Err(Error::NotImplemented {
            feature: "ILU(k) symbolic for CUDA",
        })
    }

    fn iluk_numeric(
        &self,
        _a: &CsrData<CudaRuntime>,
        _symbolic: &IlukSymbolic,
        _opts: &IlukOptions,
    ) -> Result<IlukDecomposition<CudaRuntime>> {
        Err(Error::NotImplemented {
            feature: "ILU(k) numeric for CUDA",
        })
    }

    fn iluk(
        &self,
        _a: &CsrData<CudaRuntime>,
        _opts: IlukOptions,
    ) -> Result<IlukDecomposition<CudaRuntime>> {
        Err(Error::NotImplemented {
            feature: "ILU(k) for CUDA",
        })
    }

    fn ilu0_symbolic(&self, _pattern: &CsrData<CudaRuntime>) -> Result<SymbolicIlu0> {
        Err(Error::NotImplemented {
            feature: "ILU(0) symbolic/numeric split for CUDA",
        })
    }

    fn ilu0_numeric(
        &self,
        _a: &CsrData<CudaRuntime>,
        _symbolic: &SymbolicIlu0,
        _options: IluOptions,
    ) -> Result<IluDecomposition<CudaRuntime>> {
        Err(Error::NotImplemented {
            feature: "ILU(0) symbolic/numeric split for CUDA",
        })
    }
}
