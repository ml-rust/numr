//! CPU implementation of sorting operations.

use crate::error::Result;
use crate::ops::SortingOps;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

/// SortingOps implementation for CPU runtime.
impl SortingOps<CpuRuntime> for CpuClient {
    fn sort(
        &self,
        a: &Tensor<CpuRuntime>,
        dim: isize,
        descending: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        crate::runtime::cpu::sort::sort_impl(self, a, dim, descending)
    }

    fn sort_with_indices(
        &self,
        a: &Tensor<CpuRuntime>,
        dim: isize,
        descending: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        crate::runtime::cpu::sort::sort_with_indices_impl(self, a, dim, descending)
    }

    fn argsort(
        &self,
        a: &Tensor<CpuRuntime>,
        dim: isize,
        descending: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        crate::runtime::cpu::sort::argsort_impl(self, a, dim, descending)
    }

    fn topk(
        &self,
        a: &Tensor<CpuRuntime>,
        k: usize,
        dim: isize,
        largest: bool,
        sorted: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        crate::runtime::cpu::sort::topk_impl(self, a, k, dim, largest, sorted)
    }

    fn unique(&self, a: &Tensor<CpuRuntime>, sorted: bool) -> Result<Tensor<CpuRuntime>> {
        crate::runtime::cpu::sort::unique_impl(self, a, sorted)
    }

    fn unique_with_counts(
        &self,
        a: &Tensor<CpuRuntime>,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        crate::runtime::cpu::sort::unique_with_counts_impl(self, a)
    }

    fn nonzero(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        crate::runtime::cpu::sort::nonzero_impl(self, a)
    }

    fn searchsorted(
        &self,
        sorted_sequence: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        right: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        crate::runtime::cpu::sort::searchsorted_impl(self, sorted_sequence, values, right)
    }
}
