//! CUDA implementation of convolution operations.

use crate::error::{Error, Result};
use crate::ops::{ConvOps, PaddingMode};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl ConvOps<CudaRuntime> for CudaClient {
    fn conv1d(
        &self,
        _input: &Tensor<CudaRuntime>,
        _weight: &Tensor<CudaRuntime>,
        _bias: Option<&Tensor<CudaRuntime>>,
        _stride: usize,
        _padding: PaddingMode,
        _dilation: usize,
        _groups: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        // TODO: Implement CUDA kernel for conv1d
        Err(Error::NotImplemented {
            feature: "conv1d on CUDA",
        })
    }

    fn conv2d(
        &self,
        _input: &Tensor<CudaRuntime>,
        _weight: &Tensor<CudaRuntime>,
        _bias: Option<&Tensor<CudaRuntime>>,
        _stride: (usize, usize),
        _padding: PaddingMode,
        _dilation: (usize, usize),
        _groups: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        // TODO: Implement CUDA kernel for conv2d
        Err(Error::NotImplemented {
            feature: "conv2d on CUDA",
        })
    }

    fn depthwise_conv2d(
        &self,
        _input: &Tensor<CudaRuntime>,
        _weight: &Tensor<CudaRuntime>,
        _bias: Option<&Tensor<CudaRuntime>>,
        _stride: (usize, usize),
        _padding: PaddingMode,
        _dilation: (usize, usize),
    ) -> Result<Tensor<CudaRuntime>> {
        // TODO: Implement CUDA kernel for depthwise_conv2d
        Err(Error::NotImplemented {
            feature: "depthwise_conv2d on CUDA",
        })
    }
}
