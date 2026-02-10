//! CUDA implementation of scalar operations.

use crate::error::Result;
use crate::ops::ScalarOps;
use crate::runtime::cuda::ops::helpers::native_scalar_op;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl ScalarOps<CudaRuntime> for CudaClient {
    fn add_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "add_scalar", scalar)
    }

    fn sub_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "sub_scalar", scalar)
    }

    fn mul_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "mul_scalar", scalar)
    }

    fn div_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "div_scalar", scalar)
    }

    fn pow_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "pow_scalar", scalar)
    }

    fn rsub_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "rsub_scalar", scalar)
    }
}
