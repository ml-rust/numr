//! Reduce operations for CUDA runtime
use crate::error::Result;
use crate::ops::{ReduceOps, ScalarOps};
use crate::runtime::cuda::kernels::AccumulationPrecision;
use crate::runtime::cuda::ops::helpers::native_reduce_op;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl ReduceOps<CudaRuntime> for CudaClient {
    fn sum(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "sum", dims, keepdim, None)
    }

    fn sum_with_precision(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "sum", dims, keepdim, Some(precision))
    }

    fn mean(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        // Mean = sum / count
        // When dims is empty, reduce over all dimensions
        let count: usize = if dims.is_empty() {
            a.numel()
        } else {
            dims.iter().map(|&d| a.shape()[d]).product()
        };

        // For empty dims, we need to reduce all dimensions
        let actual_dims: Vec<usize> = if dims.is_empty() {
            (0..a.shape().len()).collect()
        } else {
            dims.to_vec()
        };

        let sum_result = self.sum(a, &actual_dims, keepdim)?;
        self.div_scalar(&sum_result, count as f64)
    }

    fn max(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "max", dims, keepdim, None)
    }

    fn max_with_precision(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "max", dims, keepdim, Some(precision))
    }

    fn min(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "min", dims, keepdim, None)
    }

    fn min_with_precision(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "min", dims, keepdim, Some(precision))
    }

    fn prod(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "prod", dims, keepdim, None)
    }

    fn prod_with_precision(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
        precision: AccumulationPrecision,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "prod", dims, keepdim, Some(precision))
    }

    fn any(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "any", dims, keepdim, None)
    }

    fn all(
        &self,
        a: &Tensor<CudaRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        native_reduce_op(self, a, "all", dims, keepdim, None)
    }
}
