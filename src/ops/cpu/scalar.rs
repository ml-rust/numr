//! CPU implementation of scalar operations.

use crate::error::Result;
use crate::ops::{BinaryOp, ScalarOps};
use crate::runtime::cpu::{CpuClient, CpuRuntime, helpers::scalar_op_impl};
use crate::tensor::Tensor;

impl ScalarOps<CpuRuntime> for CpuClient {
    fn add_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Add, a, scalar, "add_scalar")
    }

    fn sub_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Sub, a, scalar, "sub_scalar")
    }

    fn mul_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Mul, a, scalar, "mul_scalar")
    }

    fn div_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Div, a, scalar, "div_scalar")
    }

    fn pow_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Pow, a, scalar, "pow_scalar")
    }
}
