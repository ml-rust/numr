//! CUDA implementation of comparison operations.

use crate::error::Result;
use crate::ops::CompareOps;
use crate::runtime::cuda::ops::helpers::native_compare_op;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl CompareOps<CudaRuntime> for CudaClient {
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
