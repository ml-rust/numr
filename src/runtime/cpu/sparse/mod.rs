//! Sparse operations implementation for CPU runtime
//!
//! This module implements the SparseOps trait for CpuRuntime, providing
//! CPU-based sparse matrix operations.

use super::{CpuClient, CpuRuntime};
use crate::error::Result;
use crate::tensor::Tensor;

// Submodules
mod dsmm;
mod esc_spgemm;
mod high_level_ops;
mod iterative;
mod linalg;
pub(crate) mod merge;
#[cfg(test)]
mod tests;

// ============================================================================
// SparseAlgorithms Trait Implementation (Backend Parity Contract)
// ============================================================================

impl crate::algorithm::sparse::SparseAlgorithms<CpuRuntime> for CpuClient {
    fn esc_spgemm_csr(
        &self,
        a_csr: &crate::sparse::CsrData<CpuRuntime>,
        b_csr: &crate::sparse::CsrData<CpuRuntime>,
    ) -> Result<crate::sparse::CsrData<CpuRuntime>> {
        esc_spgemm::esc_spgemm_csr(self, a_csr, b_csr)
    }

    fn column_parallel_dsmm(
        &self,
        dense_a: &Tensor<CpuRuntime>,
        sparse_b_csc: &crate::sparse::CscData<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        dsmm::column_parallel_dsmm(self, dense_a, sparse_b_csc)
    }
}
