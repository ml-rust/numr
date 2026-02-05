//! Sparse tensor operations for WebGPU runtime.
//!
//! This module provides GPU-accelerated sparse linear algebra operations
//! using level scheduling for parallel execution.

mod common;
mod conversions;
mod dsmm;
mod esc_spgemm;
mod high_level_ops;
mod ic0;
mod ilu0;
mod iterative;
mod merge;
mod spmv;
mod triangular_solve;

use super::{WgpuClient, WgpuRuntime};
use crate::algorithm::sparse::SparseAlgorithms;
use crate::algorithm::sparse_linalg::{
    IcDecomposition, IcOptions, IluDecomposition, IluFillLevel, IluOptions, IlukDecomposition,
    IlukOptions, IlukSymbolic, SparseLinAlgAlgorithms, SymbolicIlu0,
};
use crate::error::Error;
use crate::error::Result;
use crate::sparse::{CscData, CsrData};
use crate::tensor::Tensor;

impl SparseAlgorithms<WgpuRuntime> for WgpuClient {
    fn esc_spgemm_csr(
        &self,
        a_csr: &CsrData<WgpuRuntime>,
        b_csr: &CsrData<WgpuRuntime>,
    ) -> Result<CsrData<WgpuRuntime>> {
        esc_spgemm::esc_spgemm_csr(self, a_csr, b_csr)
    }

    fn column_parallel_dsmm(
        &self,
        dense_a: &Tensor<WgpuRuntime>,
        sparse_b_csc: &CscData<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        dsmm::column_parallel_dsmm(self, dense_a, sparse_b_csc)
    }
}

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

    fn iluk_symbolic(
        &self,
        _a: &CsrData<WgpuRuntime>,
        _level: IluFillLevel,
    ) -> Result<IlukSymbolic> {
        Err(Error::NotImplemented {
            feature: "ILU(k) symbolic for WebGPU",
        })
    }

    fn iluk_numeric(
        &self,
        _a: &CsrData<WgpuRuntime>,
        _symbolic: &IlukSymbolic,
        _opts: &IlukOptions,
    ) -> Result<IlukDecomposition<WgpuRuntime>> {
        Err(Error::NotImplemented {
            feature: "ILU(k) numeric for WebGPU",
        })
    }

    fn iluk(
        &self,
        _a: &CsrData<WgpuRuntime>,
        _opts: IlukOptions,
    ) -> Result<IlukDecomposition<WgpuRuntime>> {
        Err(Error::NotImplemented {
            feature: "ILU(k) for WebGPU",
        })
    }

    fn ilu0_symbolic(&self, _pattern: &CsrData<WgpuRuntime>) -> Result<SymbolicIlu0> {
        Err(Error::NotImplemented {
            feature: "ILU(0) symbolic/numeric split for WebGPU",
        })
    }

    fn ilu0_numeric(
        &self,
        _a: &CsrData<WgpuRuntime>,
        _symbolic: &SymbolicIlu0,
        _options: IluOptions,
    ) -> Result<IluDecomposition<WgpuRuntime>> {
        Err(Error::NotImplemented {
            feature: "ILU(0) symbolic/numeric split for WebGPU",
        })
    }
}
