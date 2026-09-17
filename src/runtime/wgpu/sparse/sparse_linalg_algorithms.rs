//! `SparseLinAlgAlgorithms` implementation for the WebGPU backend.

use super::{WgpuClient, WgpuRuntime};
use crate::algorithm::sparse_linalg::{
    IcDecomposition, IcOptions, IluDecomposition, IluFillLevel, IluOptions, IlukDecomposition,
    IlukOptions, IlukSymbolic, SparseLinAlgAlgorithms, SymbolicIlu0,
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
        super::ilu0::ilu0_wgpu(self, a, options)
    }

    fn ic0(
        &self,
        a: &CsrData<WgpuRuntime>,
        options: IcOptions,
    ) -> Result<IcDecomposition<WgpuRuntime>> {
        super::ic0::ic0_wgpu(self, a, options)
    }

    fn sparse_solve_triangular(
        &self,
        l_or_u: &CsrData<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        super::triangular_solve::sparse_solve_triangular_wgpu(self, l_or_u, b, lower, unit_diagonal)
    }

    fn iluk_symbolic(&self, a: &CsrData<WgpuRuntime>, level: IluFillLevel) -> Result<IlukSymbolic> {
        super::iluk::iluk_symbolic_wgpu(self, a, level)
    }

    fn iluk_numeric(
        &self,
        a: &CsrData<WgpuRuntime>,
        symbolic: &IlukSymbolic,
        opts: &IlukOptions,
    ) -> Result<IlukDecomposition<WgpuRuntime>> {
        super::iluk::iluk_numeric_wgpu(self, a, symbolic, opts)
    }

    fn iluk(
        &self,
        a: &CsrData<WgpuRuntime>,
        opts: IlukOptions,
    ) -> Result<IlukDecomposition<WgpuRuntime>> {
        super::iluk::iluk_wgpu(self, a, opts)
    }

    fn ilu0_symbolic(&self, pattern: &CsrData<WgpuRuntime>) -> Result<SymbolicIlu0> {
        super::ilu0::ilu0_symbolic_wgpu(self, pattern)
    }

    fn ilu0_numeric(
        &self,
        a: &CsrData<WgpuRuntime>,
        symbolic: &SymbolicIlu0,
        options: IluOptions,
    ) -> Result<IluDecomposition<WgpuRuntime>> {
        super::ilu0::ilu0_numeric_wgpu(self, a, symbolic, options)
    }
}
