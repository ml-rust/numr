//! Convolution operations for neural network layers.
//!
//! This module defines the `ConvOps` trait for 1D and 2D convolution operations
//! commonly used in neural networks for feature extraction.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Padding mode for convolution operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaddingMode {
    /// No padding - output is smaller than input.
    #[default]
    Valid,
    /// Padding to make output same size as input (when stride=1).
    Same,
    /// Custom padding specified as explicit values.
    /// For conv1d: (left, right)
    /// For conv2d: (top, bottom, left, right)
    Custom(usize, usize, usize, usize),
}

impl PaddingMode {
    /// Creates padding for a specific amount on all sides.
    pub fn uniform(padding: usize) -> Self {
        PaddingMode::Custom(padding, padding, padding, padding)
    }

    /// Creates asymmetric padding for conv1d.
    pub fn conv1d(left: usize, right: usize) -> Self {
        PaddingMode::Custom(left, right, 0, 0)
    }

    /// Creates asymmetric padding for conv2d.
    pub fn conv2d(top: usize, bottom: usize, left: usize, right: usize) -> Self {
        PaddingMode::Custom(top, bottom, left, right)
    }

    /// Returns the name of the padding mode for error messages.
    pub fn name(&self) -> &'static str {
        match self {
            PaddingMode::Valid => "valid",
            PaddingMode::Same => "same",
            PaddingMode::Custom(..) => "custom",
        }
    }
}

/// Convolution operations.
///
/// Provides 1D and 2D convolution operations commonly used in neural networks
/// for tasks like image classification, object detection, and signal processing.
///
/// # Memory Layout
///
/// All tensors use the following memory layouts:
/// - **Input 1D**: (N, C_in, L) - batch, input channels, length
/// - **Input 2D**: (N, C_in, H, W) - batch, input channels, height, width
/// - **Weight 1D**: (C_out, C_in/groups, K) - output channels, input channels per group, kernel size
/// - **Weight 2D**: (C_out, C_in/groups, K_h, K_w) - output channels, input channels per group, kernel height, kernel width
/// - **Bias**: (C_out,) - one bias per output channel
/// - **Output 1D**: (N, C_out, L_out) - batch, output channels, output length
/// - **Output 2D**: (N, C_out, H_out, W_out) - batch, output channels, output height, output width
///
/// # Backend Support
///
/// ## Data Types
///
/// - **CPU**: Supports F32, F64, F16, BF16 (with `f16` feature)
/// - **CUDA**: Supports F32, F64, F16, BF16 (with `f16` feature)
/// - **WebGPU**: Currently supports F32 only
///
/// All backends require floating-point dtypes. Integer dtypes are not supported.
pub trait ConvOps<R: Runtime> {
    /// Applies a 1D convolution over an input signal.
    ///
    /// Given input of shape (N, C_in, L) and weight of shape (C_out, C_in/groups, K),
    /// produces output of shape (N, C_out, L_out).
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (N, C_in, L)
    /// * `weight` - Convolution kernel of shape (C_out, C_in/groups, K)
    /// * `bias` - Optional bias of shape (C_out,)
    /// * `stride` - Stride of the convolution (default: 1)
    /// * `padding` - Padding mode
    /// * `dilation` - Spacing between kernel elements (default: 1)
    /// * `groups` - Number of blocked connections from input to output channels (default: 1)
    ///
    /// # Returns
    ///
    /// Output tensor of shape (N, C_out, L_out) where:
    /// L_out = floor((L + pad_left + pad_right - dilation * (K - 1) - 1) / stride + 1)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - Input is not a 3D tensor
    /// - Weight is not a 3D tensor
    /// - Bias is not a 1D tensor with length C_out
    /// - C_in is not divisible by groups
    /// - C_out is not divisible by groups
    /// - stride, dilation, or groups is 0
    ///
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Input: batch=2, channels=3, length=10
    /// let input = Tensor::randn(&[2, 3, 10], DType::F32, &device)?;
    /// // Kernel: 16 output channels, 3 input channels, kernel size 3
    /// let weight = Tensor::randn(&[16, 3, 3], DType::F32, &device)?;
    /// let bias = Tensor::zeros(&[16], DType::F32, &device)?;
    ///
    /// let output = client.conv1d(&input, &weight, Some(&bias), 1, PaddingMode::Same, 1, 1)?;
    /// // output has shape (2, 16, 10)
    /// ```
    fn conv1d(
        &self,
        input: &Tensor<R>,
        weight: &Tensor<R>,
        bias: Option<&Tensor<R>>,
        stride: usize,
        padding: PaddingMode,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor<R>>;

    /// Applies a 2D convolution over an input image.
    ///
    /// Given input of shape (N, C_in, H, W) and weight of shape (C_out, C_in/groups, K_h, K_w),
    /// produces output of shape (N, C_out, H_out, W_out).
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (N, C_in, H, W)
    /// * `weight` - Convolution kernel of shape (C_out, C_in/groups, K_h, K_w)
    /// * `bias` - Optional bias of shape (C_out,)
    /// * `stride` - Stride of the convolution as (stride_h, stride_w)
    /// * `padding` - Padding mode
    /// * `dilation` - Spacing between kernel elements as (dilation_h, dilation_w)
    /// * `groups` - Number of blocked connections from input to output channels
    ///
    /// # Returns
    ///
    /// Output tensor of shape (N, C_out, H_out, W_out) where:
    /// H_out = floor((H + pad_top + pad_bottom - dilation_h * (K_h - 1) - 1) / stride_h + 1)
    /// W_out = floor((W + pad_left + pad_right - dilation_w * (K_w - 1) - 1) / stride_w + 1)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - Input is not a 4D tensor
    /// - Weight is not a 4D tensor
    /// - Bias is not a 1D tensor with length C_out
    /// - C_in is not divisible by groups
    /// - C_out is not divisible by groups
    /// - Any stride, dilation, or groups value is 0
    ///
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Input: batch=2, channels=3, height=32, width=32
    /// let input = Tensor::randn(&[2, 3, 32, 32], DType::F32, &device)?;
    /// // Kernel: 64 output channels, 3 input channels, 3x3 kernel
    /// let weight = Tensor::randn(&[64, 3, 3, 3], DType::F32, &device)?;
    /// let bias = Tensor::zeros(&[64], DType::F32, &device)?;
    ///
    /// let output = client.conv2d(&input, &weight, Some(&bias), (1, 1), PaddingMode::Same, (1, 1), 1)?;
    /// // output has shape (2, 64, 32, 32)
    /// ```
    fn conv2d(
        &self,
        input: &Tensor<R>,
        weight: &Tensor<R>,
        bias: Option<&Tensor<R>>,
        stride: (usize, usize),
        padding: PaddingMode,
        dilation: (usize, usize),
        groups: usize,
    ) -> Result<Tensor<R>>;

    /// Applies a depthwise separable 2D convolution.
    ///
    /// In depthwise convolution, each input channel is convolved separately with
    /// its own set of filters. This is equivalent to grouped convolution where
    /// groups = C_in = C_out.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (N, C, H, W)
    /// * `weight` - Depthwise kernel of shape (C, 1, K_h, K_w)
    /// * `bias` - Optional bias of shape (C,)
    /// * `stride` - Stride of the convolution as (stride_h, stride_w)
    /// * `padding` - Padding mode
    /// * `dilation` - Spacing between kernel elements as (dilation_h, dilation_w)
    ///
    /// # Returns
    ///
    /// Output tensor of shape (N, C, H_out, W_out).
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - Input is not a 4D tensor
    /// - Weight is not a 4D tensor with shape[1] = 1
    /// - Weight channels don't match input channels
    /// - Bias is not a 1D tensor with length C
    /// - Any stride or dilation value is 0
    ///
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Input: batch=2, channels=32, height=28, width=28
    /// let input = Tensor::randn(&[2, 32, 28, 28], DType::F32, &device)?;
    /// // Depthwise kernel: 32 channels, 1 input per group, 3x3 kernel
    /// let weight = Tensor::randn(&[32, 1, 3, 3], DType::F32, &device)?;
    ///
    /// let output = client.depthwise_conv2d(&input, &weight, None, (1, 1), PaddingMode::Same, (1, 1))?;
    /// // output has shape (2, 32, 28, 28)
    /// ```
    fn depthwise_conv2d(
        &self,
        input: &Tensor<R>,
        weight: &Tensor<R>,
        bias: Option<&Tensor<R>>,
        stride: (usize, usize),
        padding: PaddingMode,
        dilation: (usize, usize),
    ) -> Result<Tensor<R>>;
}
