//! Fused 2D convolution: conv + bias + activation + scaled residual.
//!
//! One call runs the whole conv block of a super-resolution network. The
//! `_into` variant writes into a view of a larger buffer, such as a channel
//! slice of a dense-block feature buffer.

use std::fmt;

use super::PaddingMode;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Memory layout of a fused convolution's input, weight and output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvLayout {
    /// Input `[N, C, H, W]`, weight `[C_out, C_in/groups, Kh, Kw]` (OIHW),
    /// output `[N, C_out, H_out, W_out]`.
    Nchw,
    /// Input `[N, H, W, C]`, weight `[C_out, Kh, Kw, C_in/groups]` (OHWI),
    /// output `[N, H_out, W_out, C_out]`.
    Nhwc,
}

/// Activation applied after the bias add.
pub enum ConvActivation<'a, R: Runtime> {
    /// Identity.
    None,
    /// `max(0, y)`.
    Relu,
    /// `y` if `y > 0`, else `slope * y`. The slope must be finite.
    LeakyRelu(f64),
    /// `y` if `y > 0`, else `weight[c] * y` for output channel `c`.
    ///
    /// The weight has shape `[C_out]` or `[1]` and the input's dtype. A `[1]`
    /// weight is one slope shared by every channel.
    Prelu(&'a Tensor<R>),
}

impl<R: Runtime> Clone for ConvActivation<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime> Copy for ConvActivation<'_, R> {}

impl<R: Runtime> fmt::Debug for ConvActivation<'_, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("None"),
            Self::Relu => f.write_str("Relu"),
            Self::LeakyRelu(slope) => f.debug_tuple("LeakyRelu").field(slope).finish(),
            Self::Prelu(weight) => f.debug_tuple("Prelu").field(weight).finish(),
        }
    }
}

/// Scaled residual added last: `out = tensor + scale * y`.
///
/// `tensor` has the output's shape and dtype, in the output's layout. It can
/// be a strided view. `scale` must be finite.
pub struct ConvResidual<'a, R: Runtime> {
    /// Tensor added to the scaled conv result.
    pub tensor: &'a Tensor<R>,
    /// Factor applied to the conv result before the add.
    pub scale: f64,
}

impl<R: Runtime> Clone for ConvResidual<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime> Copy for ConvResidual<'_, R> {}

impl<R: Runtime> fmt::Debug for ConvResidual<'_, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConvResidual")
            .field("tensor", self.tensor)
            .field("scale", &self.scale)
            .finish()
    }
}

/// Accumulator precision a fused convolution can use for its K sum.
///
/// On the native CUDA path the bias, activation and residual run in F32
/// under either choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ConvAccumulation {
    /// Accumulate in F32, or in F64 for F64 inputs.
    #[default]
    F32,
    /// Permit the K sum in the input's half dtype.
    ///
    /// This is a permission. A backend can ignore it and accumulate in F32.
    /// Only the native CUDA path honours it, and only for F16 inputs on
    /// sm_75+. BF16 always accumulates in F32.
    ///
    /// The result can differ from F32 accumulation. The difference grows
    /// with `K = Kh * Kw * C_in / groups`. A partial sum past the F16 range
    /// becomes `inf`. A fixed input still gives a bit-identical output on
    /// every run.
    AllowHalf,
}

/// Geometry and accumulator precision of a fused 2D convolution.
///
/// [`Conv2dConfig::new`] sets stride 1, `Valid` padding, dilation 1, one
/// group and [`ConvAccumulation::F32`]. The `with_*` setters change one
/// field each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Conv2dConfig {
    /// Stride as `(stride_h, stride_w)`. Each must be positive.
    pub stride: (usize, usize),
    /// Padding mode. `Custom` is `(top, bottom, left, right)`.
    pub padding: PaddingMode,
    /// Dilation as `(dilation_h, dilation_w)`. Each must be positive.
    pub dilation: (usize, usize),
    /// Channel groups. `C_in` and `C_out` must be multiples of it.
    pub groups: usize,
    /// Layout of input, weight, output and residual.
    pub layout: ConvLayout,
    /// Accumulator precision of the K sum. See [`ConvAccumulation`].
    pub accumulation: ConvAccumulation,
}

impl Conv2dConfig {
    /// Stride 1, `Valid` padding, dilation 1, one group, F32 accumulation,
    /// in `layout`.
    pub fn new(layout: ConvLayout) -> Self {
        Self {
            stride: (1, 1),
            padding: PaddingMode::Valid,
            dilation: (1, 1),
            groups: 1,
            layout,
            accumulation: ConvAccumulation::F32,
        }
    }

    /// Set the stride as `(stride_h, stride_w)`.
    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding mode.
    pub fn with_padding(mut self, padding: PaddingMode) -> Self {
        self.padding = padding;
        self
    }

    /// Set the dilation as `(dilation_h, dilation_w)`.
    pub fn with_dilation(mut self, dilation: (usize, usize)) -> Self {
        self.dilation = dilation;
        self
    }

    /// Set the number of channel groups.
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    /// Set the accumulator precision. See [`ConvAccumulation`].
    pub fn with_accumulation(mut self, accumulation: ConvAccumulation) -> Self {
        self.accumulation = accumulation;
        self
    }
}

/// Fused 2D convolution with bias, activation and scaled residual.
///
/// # Math
///
/// The order is fixed:
///
/// ```text
/// y = conv(x, w) + bias
/// y = act(y)
/// if residual: y = residual + scale * y
/// ```
///
/// # Layouts and shapes
///
/// | Layout | Input | Weight | Output and residual |
/// | --- | --- | --- | --- |
/// | `Nchw` | `[N, C_in, H, W]` | `[C_out, C_in/groups, Kh, Kw]` | `[N, C_out, H_out, W_out]` |
/// | `Nhwc` | `[N, H, W, C_in]` | `[C_out, Kh, Kw, C_in/groups]` | `[N, H_out, W_out, C_out]` |
///
/// `bias` is `[C_out]`. A `Prelu` weight is `[C_out]` or `[1]`. `H_out` and
/// `W_out` follow [`ConvOps::conv2d`](super::ConvOps::conv2d). Any operand can
/// be a strided view.
///
/// # Dtypes
///
/// Every operand has the input's float dtype. Each backend supports the
/// dtypes its [`ConvOps::conv2d`](super::ConvOps::conv2d) supports. WebGPU
/// supports F32 only.
///
/// # Accumulation
///
/// The K sum runs in F32 unless [`Conv2dConfig::accumulation`] permits half
/// accumulation and the backend honours it. See [`ConvAccumulation`].
///
/// # Autograd
///
/// Inference only. No gradient is recorded.
pub trait FusedConvOps<R: Runtime> {
    /// Run the fused convolution and return a new contiguous tensor.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` for a non-4D input or weight, a bad
    ///   stride, dilation or group count, a channel mismatch, a bias of the
    ///   wrong length, a `Prelu` weight that is not `[C_out]` or `[1]`, or a
    ///   non-finite slope or scale.
    /// - `Error::ShapeMismatch` when the residual shape differs from the output shape.
    /// - `Error::DTypeMismatch` when a `Prelu` weight or residual has another dtype.
    /// - `Error::UnsupportedDType` for a non-float input.
    fn conv2d_fused(
        &self,
        input: &Tensor<R>,
        weight: &Tensor<R>,
        bias: Option<&Tensor<R>>,
        config: &Conv2dConfig,
        activation: ConvActivation<'_, R>,
        residual: Option<ConvResidual<'_, R>>,
    ) -> Result<Tensor<R>>;

    /// Run the fused convolution and write the result into `out`.
    ///
    /// `out` has the output shape and the input's dtype. It can be a strided
    /// view, such as `buf.narrow(-1, 64, 32)` of an NHWC buffer or
    /// `buf.narrow(1, 64, 32)` of an NCHW buffer. Elements of the parent
    /// buffer outside `out` keep their values. `out` must not repeat an
    /// element (stride 0 on a dimension larger than 1).
    ///
    /// # Aliasing
    ///
    /// `out` must not share an element with `input`, `weight`, `bias`, the
    /// `Prelu` weight or the residual. A view of the same storage passes when
    /// it is provably disjoint:
    ///
    /// - Exact rule: both views have equal strides and equal shapes except
    ///   along one dimension. The offset difference is a whole multiple of
    ///   that dimension's stride, the two index ranges along it do not
    ///   intersect, and the combined view addresses no element twice.
    /// - Fallback rule: the element spans of the two views do not intersect.
    ///   See [`BinaryOps::copy_into`](super::BinaryOps::copy_into).
    ///
    /// The exact rule accepts a dense block that reads channels `[0, k)` of
    /// one buffer and writes channels `[k, k + g)` of the same buffer. An
    /// identical view counts as overlap.
    ///
    /// # Errors
    ///
    /// Every error of [`conv2d_fused`](Self::conv2d_fused), plus:
    /// - `Error::ShapeMismatch` when `out` has the wrong shape.
    /// - `Error::DTypeMismatch` when `out` has another dtype.
    /// - `Error::InvalidArgument` for argument `out` when it repeats an
    ///   element or overlaps a read operand. The reason names the operand.
    fn conv2d_fused_into(
        &self,
        out: &Tensor<R>,
        input: &Tensor<R>,
        weight: &Tensor<R>,
        bias: Option<&Tensor<R>>,
        config: &Conv2dConfig,
        activation: ConvActivation<'_, R>,
        residual: Option<ConvResidual<'_, R>>,
    ) -> Result<()>;
}
