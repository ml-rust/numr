//! Reduction operations for WebGPU runtime

use crate::error::Result;
use crate::ops::{AccumulationPrecision, ReduceOps};
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::native::native_reduce_op;
use crate::tensor::Tensor;

impl ReduceOps<WgpuRuntime> for WgpuClient {
    fn sum(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_reduce_op(self, "sum", a, dims, keepdim)
    }

    fn sum_with_precision(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        _precision: AccumulationPrecision,
    ) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU doesn't support accumulation precision control
        // Just delegate to standard sum
        native_reduce_op(self, "sum", a, dims, keepdim)
    }

    fn mean(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_reduce_op(self, "mean", a, dims, keepdim)
    }

    fn max(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_reduce_op(self, "max", a, dims, keepdim)
    }

    fn max_with_precision(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        _precision: AccumulationPrecision,
    ) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU doesn't support accumulation precision control
        // Just delegate to standard max
        native_reduce_op(self, "max", a, dims, keepdim)
    }

    fn min(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_reduce_op(self, "min", a, dims, keepdim)
    }

    fn min_with_precision(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        _precision: AccumulationPrecision,
    ) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU doesn't support accumulation precision control
        // Just delegate to standard min
        native_reduce_op(self, "min", a, dims, keepdim)
    }

    fn prod(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_reduce_op(self, "prod", a, dims, keepdim)
    }

    fn prod_with_precision(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
        _precision: AccumulationPrecision,
    ) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU doesn't support accumulation precision control
        // Just delegate to standard prod
        native_reduce_op(self, "prod", a, dims, keepdim)
    }

    fn any(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_reduce_op(self, "any", a, dims, keepdim)
    }

    fn all(
        &self,
        a: &Tensor<WgpuRuntime>,
        dims: &[usize],
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_reduce_op(self, "all", a, dims, keepdim)
    }
}
