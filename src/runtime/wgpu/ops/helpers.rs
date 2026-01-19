//! Helper functions and parameter structs for WebGPU operations.
//!
//! This module provides shared utilities for operation implementation including
//! buffer creation, tensor access, and parameter struct definitions.

use wgpu::BufferUsages;

use super::super::WgpuRuntime;
use super::super::client::get_buffer;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Create a uniform buffer with the given data.
pub(super) fn create_params_buffer<T: bytemuck::Pod>(
    client: &super::super::WgpuClient,
    data: &T,
) -> wgpu::Buffer {
    let buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: std::mem::size_of::<T>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .queue
        .write_buffer(&buffer, 0, bytemuck::bytes_of(data));
    buffer
}

/// Get the wgpu buffer from a tensor's storage pointer.
pub(super) fn get_tensor_buffer(
    tensor: &Tensor<WgpuRuntime>,
) -> Result<std::sync::Arc<wgpu::Buffer>> {
    let ptr = tensor.storage().ptr();
    get_buffer(ptr).ok_or_else(|| Error::Internal("Buffer not found in registry".to_string()))
}

/// Allocate output tensor with given shape and dtype.
pub(super) fn alloc_output(
    client: &super::super::WgpuClient,
    shape: &[usize],
    dtype: DType,
) -> Tensor<WgpuRuntime> {
    Tensor::empty(shape, dtype, client.device())
}

// ============================================================================
// Params Structs (must match WGSL shader structs)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BinaryParams {
    pub(super) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct UnaryParams {
    pub(super) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ScalarParams {
    pub(super) numel: u32,
    pub(super) scalar: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ClampParams {
    pub(super) numel: u32,
    pub(super) min_val: f32,
    pub(super) max_val: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct WhereParams {
    pub(super) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CastParams {
    pub(super) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ReduceParams {
    pub(super) reduce_size: u32,
    pub(super) outer_size: u32,
    pub(super) inner_size: u32,
    pub(super) numel_out: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct FullReduceParams {
    pub(super) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SoftmaxParams {
    pub(super) batch_size: u32,
    pub(super) dim_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ArgReduceParams {
    pub(super) reduce_size: u32,
    pub(super) outer_size: u32,
    pub(super) inner_size: u32,
    pub(super) numel_out: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MatmulParams {
    pub(super) m: u32,
    pub(super) k: u32,
    pub(super) n: u32,
    pub(super) batch_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct RmsNormParams {
    pub(super) batch_size: u32,
    pub(super) hidden_size: u32,
    pub(super) eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LayerNormParams {
    pub(super) batch_size: u32,
    pub(super) hidden_size: u32,
    pub(super) eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CatShaderParams {
    pub(super) outer_size: u32,
    pub(super) src_cat_size: u32,
    pub(super) dst_cat_size: u32,
    pub(super) cat_offset: u32,
    pub(super) inner_size: u32,
    pub(super) total_elements: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct IndexSelectParams {
    pub(super) outer_size: u32,
    pub(super) dim_size: u32,
    pub(super) inner_size: u32,
    pub(super) index_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GatherParams {
    pub(super) ndim: u32,
    pub(super) dim: u32,
    pub(super) total_elements: u32,
    pub(super) _padding: u32,
    pub(super) input_shape: [u32; 4],
    pub(super) input_strides: [u32; 4],
    pub(super) output_shape: [u32; 4],
    pub(super) output_strides: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ScatterParams {
    pub(super) ndim: u32,
    pub(super) dim: u32,
    pub(super) src_total: u32,
    pub(super) _padding: u32,
    pub(super) output_shape: [u32; 4],
    pub(super) output_strides: [u32; 4],
    pub(super) src_shape: [u32; 4],
    pub(super) src_strides: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CopyParams {
    pub(super) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MaskedFillParams {
    pub(super) numel: u32,
    pub(super) fill_value: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MaskedCountParams {
    pub(super) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MaskedSelectParams {
    pub(super) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ArangeParams {
    pub(super) numel: u32,
    pub(super) start: f32,
    pub(super) step: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LinspaceParams {
    pub(super) steps: u32,
    pub(super) start: f32,
    pub(super) stop: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct EyeParams {
    pub(super) n: u32,
    pub(super) m: u32,
    pub(super) numel: u32,
}
