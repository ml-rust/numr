//! CPU implementation of shape operations.

use crate::error::Result;
use crate::ops::ShapeOps;
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{cat_impl, chunk_impl, pad_impl, repeat_impl, roll_impl, split_impl, stack_impl},
};
use crate::tensor::Tensor;

/// ShapeOps implementation for CPU runtime.
impl ShapeOps<CpuRuntime> for CpuClient {
    fn cat(&self, tensors: &[&Tensor<CpuRuntime>], dim: isize) -> Result<Tensor<CpuRuntime>> {
        cat_impl(self, tensors, dim)
    }

    fn stack(&self, tensors: &[&Tensor<CpuRuntime>], dim: isize) -> Result<Tensor<CpuRuntime>> {
        stack_impl(self, tensors, dim)
    }

    fn split(
        &self,
        tensor: &Tensor<CpuRuntime>,
        split_size: usize,
        dim: isize,
    ) -> Result<Vec<Tensor<CpuRuntime>>> {
        split_impl(tensor, split_size, dim)
    }

    fn chunk(
        &self,
        tensor: &Tensor<CpuRuntime>,
        chunks: usize,
        dim: isize,
    ) -> Result<Vec<Tensor<CpuRuntime>>> {
        chunk_impl(tensor, chunks, dim)
    }

    fn repeat(&self, tensor: &Tensor<CpuRuntime>, repeats: &[usize]) -> Result<Tensor<CpuRuntime>> {
        repeat_impl(self, tensor, repeats)
    }

    fn pad(
        &self,
        tensor: &Tensor<CpuRuntime>,
        padding: &[usize],
        value: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        pad_impl(self, tensor, padding, value)
    }

    fn roll(
        &self,
        tensor: &Tensor<CpuRuntime>,
        shift: isize,
        dim: isize,
    ) -> Result<Tensor<CpuRuntime>> {
        roll_impl(self, tensor, shift, dim)
    }
}
