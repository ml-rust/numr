//! CPU implementation of reduce operations.

use crate::error::Result;
use crate::ops::{AccumulationPrecision, ReduceOp, ReduceOps};
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{reduce_impl, reduce_impl_with_precision},
};
use crate::tensor::Tensor;

/// ReduceOps implementation for CPU runtime.
impl ReduceOps<CpuRuntime> for CpuClient {
    fn sum(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl(self, ReduceOp::Sum, a, dims, keepdim, "sum")
    }

    fn sum_with_precision(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl_with_precision(self, ReduceOp::Sum, a, dims, keepdim, precision, "sum")
    }

    fn mean(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl(self, ReduceOp::Mean, a, dims, keepdim, "mean")
    }

    fn max(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl(self, ReduceOp::Max, a, dims, keepdim, "max")
    }

    fn max_with_precision(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl_with_precision(self, ReduceOp::Max, a, dims, keepdim, precision, "max")
    }

    fn min(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl(self, ReduceOp::Min, a, dims, keepdim, "min")
    }

    fn min_with_precision(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl_with_precision(self, ReduceOp::Min, a, dims, keepdim, precision, "min")
    }

    fn prod(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl(self, ReduceOp::Prod, a, dims, keepdim, "prod")
    }

    fn prod_with_precision(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl_with_precision(self, ReduceOp::Prod, a, dims, keepdim, precision, "prod")
    }

    fn any(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl(self, ReduceOp::Any, a, dims, keepdim, "any")
    }

    fn all(
        &self,
        a: &Tensor<CpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        reduce_impl(self, ReduceOp::All, a, dims, keepdim, "all")
    }
}
