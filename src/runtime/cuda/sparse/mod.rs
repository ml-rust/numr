//! Sparse operations implementation for CUDA runtime
//!
//! This module implements the SparseOps trait for CudaRuntime, providing
//! GPU-accelerated sparse matrix operations using CUDA kernels.
//!
//! ## Module Structure
//!
//! - `spmv` - SpMV, SpMM and diagonal extraction
//! - `merge` - Element-wise operations (add/sub/mul/div) for CSR/CSC/COO formats
//! - `conversions` - Format conversions (COO↔CSR, COO↔CSC, CSR↔CSC)
//! - `formats` - GPU format selection for the high-level operations
//! - `indices` - I64 ↔ I32 index casts for the `int` kernels

// Submodules
mod conversions;
mod dsmm;
mod esc_spgemm;
mod formats;
mod high_level_ops;
mod indices;
mod iterative;
mod linalg;
mod merge;
mod spmv;

// ============================================================================
// SparseAlgorithms Trait Implementation (Backend Parity Contract)
// ============================================================================

use super::{CudaClient, CudaRuntime};
use crate::error::Result;
use crate::sparse::{CscData, CsrData};
use crate::tensor::Tensor;

impl crate::algorithm::sparse::SparseAlgorithms<CudaRuntime> for CudaClient {
    fn esc_spgemm_csr(
        &self,
        a_csr: &CsrData<CudaRuntime>,
        b_csr: &CsrData<CudaRuntime>,
    ) -> Result<CsrData<CudaRuntime>> {
        esc_spgemm::esc_spgemm_csr(self, a_csr, b_csr)
    }

    fn column_parallel_dsmm(
        &self,
        dense_a: &Tensor<CudaRuntime>,
        sparse_b_csc: &CscData<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        dsmm::column_parallel_dsmm(self, dense_a, sparse_b_csc)
    }
}
