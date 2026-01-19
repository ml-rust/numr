//! CompareOps trait implementation for WebGPU runtime.

use super::super::{WgpuClient, WgpuRuntime};
use super::native::*;
use crate::error::Result;
use crate::ops::CompareOps;
use crate::tensor::Tensor;

impl CompareOps<WgpuRuntime> for WgpuClient {
    fn eq(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_compare_op(self, "eq", a, b)
    }

    fn ne(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_compare_op(self, "ne", a, b)
    }

    fn lt(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_compare_op(self, "lt", a, b)
    }

    fn le(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_compare_op(self, "le", a, b)
    }

    fn gt(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_compare_op(self, "gt", a, b)
    }

    fn ge(&self, a: &Tensor<WgpuRuntime>, b: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_compare_op(self, "ge", a, b)
    }
}
