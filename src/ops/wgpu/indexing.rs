//! Indexing operations for WebGPU runtime

use crate::error::{Error, Result};
use crate::ops::{IndexingOps, ScatterReduceOp};
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::native::{
    native_argreduce_op, native_embedding_lookup, native_gather, native_index_put,
    native_index_select, native_masked_fill, native_masked_select, native_scatter,
};
use crate::tensor::Tensor;

impl IndexingOps<WgpuRuntime> for WgpuClient {
    fn argmax(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_argreduce_op(self, "argmax", a, dim, keepdim)
    }

    fn argmin(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        keepdim: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_argreduce_op(self, "argmin", a, dim, keepdim)
    }

    fn gather(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        index: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_gather(self, a, dim, index)
    }

    fn scatter(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        index: &Tensor<WgpuRuntime>,
        src: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_scatter(self, a, dim, index, src)
    }

    fn index_select(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        index: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_index_select(self, a, dim, index)
    }

    fn index_put(
        &self,
        a: &Tensor<WgpuRuntime>,
        dim: usize,
        index: &Tensor<WgpuRuntime>,
        src: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_index_put(self, a, dim, index, src)
    }

    fn masked_select(
        &self,
        a: &Tensor<WgpuRuntime>,
        mask: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_masked_select(self, a, mask)
    }

    fn masked_fill(
        &self,
        a: &Tensor<WgpuRuntime>,
        mask: &Tensor<WgpuRuntime>,
        value: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_masked_fill(self, a, mask, value)
    }

    fn embedding_lookup(
        &self,
        embeddings: &Tensor<WgpuRuntime>,
        indices: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_embedding_lookup(self, embeddings, indices)
    }

    fn scatter_reduce(
        &self,
        _dst: &Tensor<WgpuRuntime>,
        _dim: usize,
        _index: &Tensor<WgpuRuntime>,
        _src: &Tensor<WgpuRuntime>,
        _op: ScatterReduceOp,
        _include_self: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        // TODO: Implement WebGPU shader for scatter_reduce
        // Requires atomics for reduction operations (Sum, Mean) or compare-and-swap for Max/Min
        Err(Error::NotImplemented {
            feature: "scatter_reduce on WebGPU (requires atomic operations)",
        })
    }

    fn gather_nd(
        &self,
        _input: &Tensor<WgpuRuntime>,
        _indices: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        // TODO: Implement WebGPU shader for gather_nd
        Err(Error::NotImplemented {
            feature: "gather_nd on WebGPU",
        })
    }

    fn bincount(
        &self,
        _input: &Tensor<WgpuRuntime>,
        _weights: Option<&Tensor<WgpuRuntime>>,
        _minlength: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        // TODO: Implement WebGPU shader for bincount
        // Requires atomics for counting
        Err(Error::NotImplemented {
            feature: "bincount on WebGPU (requires atomic operations)",
        })
    }
}
