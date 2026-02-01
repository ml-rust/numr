//! CPU implementation of binary operations.

use crate::error::Result;
use crate::ops::BinaryOps;
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{BinaryOp, binary_op_impl},
};
use crate::tensor::Tensor;

/// BinaryOps implementation for CPU runtime.
impl BinaryOps<CpuRuntime> for CpuClient {
    fn add(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Add, a, b, "add")
    }

    fn sub(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Sub, a, b, "sub")
    }

    fn mul(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Mul, a, b, "mul")
    }

    fn div(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Div, a, b, "div")
    }

    fn pow(&self, a: &Tensor<CpuRuntime>, b: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Pow, a, b, "pow")
    }

    fn maximum(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Max, a, b, "maximum")
    }

    fn minimum(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Min, a, b, "minimum")
    }

    fn atan2(&self, y: &Tensor<CpuRuntime>, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_op_impl(self, BinaryOp::Atan2, y, x, "atan2")
    }
}
