//! WebGPU implementation of convolution operations.

use crate::error::{Error, Result};
use crate::ops::{ConvOps, PaddingMode};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

impl ConvOps<WgpuRuntime> for WgpuClient {
    fn conv1d(
        &self,
        _input: &Tensor<WgpuRuntime>,
        _weight: &Tensor<WgpuRuntime>,
        _bias: Option<&Tensor<WgpuRuntime>>,
        _stride: usize,
        _padding: PaddingMode,
        _dilation: usize,
        _groups: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        // TODO: Implement WebGPU shader for conv1d
        Err(Error::NotImplemented {
            feature: "conv1d on WebGPU",
        })
    }

    fn conv2d(
        &self,
        _input: &Tensor<WgpuRuntime>,
        _weight: &Tensor<WgpuRuntime>,
        _bias: Option<&Tensor<WgpuRuntime>>,
        _stride: (usize, usize),
        _padding: PaddingMode,
        _dilation: (usize, usize),
        _groups: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        // TODO: Implement WebGPU shader for conv2d
        Err(Error::NotImplemented {
            feature: "conv2d on WebGPU",
        })
    }

    fn depthwise_conv2d(
        &self,
        _input: &Tensor<WgpuRuntime>,
        _weight: &Tensor<WgpuRuntime>,
        _bias: Option<&Tensor<WgpuRuntime>>,
        _stride: (usize, usize),
        _padding: PaddingMode,
        _dilation: (usize, usize),
    ) -> Result<Tensor<WgpuRuntime>> {
        // TODO: Implement WebGPU shader for depthwise_conv2d
        Err(Error::NotImplemented {
            feature: "depthwise_conv2d on WebGPU",
        })
    }
}
