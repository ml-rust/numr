//! CPU implementation of comparison operations.

use crate::error::Result;
use crate::ops::{CompareOp, CompareOps};
use crate::runtime::cpu::{CpuClient, CpuRuntime, helpers::compare_op_impl};
use crate::tensor::Tensor;

impl CompareOps<CpuRuntime> for CpuClient {
    fn eq(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, CompareOp::Eq, a, b, "eq")
    }

    fn ne(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, CompareOp::Ne, a, b, "ne")
    }

    fn lt(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, CompareOp::Lt, a, b, "lt")
    }

    fn le(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, CompareOp::Le, a, b, "le")
    }

    fn gt(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, CompareOp::Gt, a, b, "gt")
    }

    fn ge(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        compare_op_impl(self, CompareOp::Ge, a, b, "ge")
    }
}
