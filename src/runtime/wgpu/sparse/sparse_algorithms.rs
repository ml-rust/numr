//! `SparseAlgorithms` implementation for the WebGPU backend.

use super::{WgpuClient, WgpuRuntime};
use crate::algorithm::sparse::SparseAlgorithms;
use crate::error::Result;
use crate::sparse::{CscData, CsrData};
use crate::tensor::Tensor;

impl SparseAlgorithms<WgpuRuntime> for WgpuClient {
    fn esc_spgemm_csr(
        &self,
        a_csr: &CsrData<WgpuRuntime>,
        b_csr: &CsrData<WgpuRuntime>,
    ) -> Result<CsrData<WgpuRuntime>> {
        super::esc_spgemm::esc_spgemm_csr(self, a_csr, b_csr)
    }

    fn column_parallel_dsmm(
        &self,
        dense_a: &Tensor<WgpuRuntime>,
        sparse_b_csc: &CscData<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        super::dsmm::column_parallel_dsmm(self, dense_a, sparse_b_csc)
    }
}
