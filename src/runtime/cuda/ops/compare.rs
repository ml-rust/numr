//! CompareOps implementation for CUDA runtime

use super::super::CudaRuntime;
use super::helpers::native_compare_op;
use crate::error::Result;
use crate::ops::CompareOps;
use crate::tensor::Tensor;

impl CompareOps<CudaRuntime> for crate::runtime::cuda::CudaClient {
    fn eq(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_compare_op(self, a, b, "eq")
    }

    fn ne(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_compare_op(self, a, b, "ne")
    }

    fn lt(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_compare_op(self, a, b, "lt")
    }

    fn le(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_compare_op(self, a, b, "le")
    }

    fn gt(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_compare_op(self, a, b, "gt")
    }

    fn ge(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_compare_op(self, a, b, "ge")
    }
}
