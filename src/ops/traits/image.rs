//! Image data-movement operations: pixel shuffle, pixel unshuffle and
//! nearest-neighbour upsampling.
//!
//! Each op moves elements and computes nothing, so every backend covers every
//! dtype it stores. The `_into` variants write into a caller-given view, such
//! as a channel slice of a larger buffer.

use super::ConvLayout;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Output geometry of [`ImageOps::upsample_nearest2d`].
///
/// Both forms give `(height, width)` in that order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NearestSize {
    /// Scale factors `(sh, sw)`. Each must be finite and positive.
    ///
    /// The output size is `floor(in * s)` in f64, which must be at least 1.
    /// The source index uses `scale = 1 / s`, rounded to f32.
    Scale(f64, f64),
    /// Output size `(oh, ow)`. Each must be at least 1.
    ///
    /// The source index uses `scale = in / out`, divided in f32.
    Size(usize, usize),
}

/// Pixel shuffle, pixel unshuffle and nearest upsampling on 4-D image tensors.
///
/// # Layouts
///
/// `layout` names the dimension order of the input and the output:
///
/// | Layout | Tensor shape |
/// | --- | --- |
/// | `ConvLayout::Nchw` | `[N, C, H, W]` |
/// | `ConvLayout::Nhwc` | `[N, H, W, C]` |
///
/// Only the dimension order differs. An NHWC call returns the NCHW result
/// with its dimensions permuted to NHWC.
///
/// # Dtypes
///
/// CPU and CUDA support every dtype. WebGPU supports F32, I32 and U32.
///
/// # Strided views
///
/// `input` can be any strided view. `out` of an `_into` call can be any
/// strided view that addresses each element once.
///
/// # Autograd
///
/// No gradient is recorded.
pub trait ImageOps<R: Runtime> {
    /// Rearrange `[N, C·r², H, W]` into `[N, C, H·r, W·r]`, matching
    /// `torch.nn.functional.pixel_shuffle`.
    ///
    /// With `r = factor`, `out[n, c, h·r + i, w·r + j] = in[n, c·r² + i·r + j, h, w]`.
    /// In NHWC, `in[n, h, w, c·r² + i·r + j]` moves to `out[n, h·r + i, w·r + j, c]`.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` for argument `input` when the rank is not 4,
    ///   or when the channel count is not divisible by `factor²`.
    /// - `Error::InvalidArgument` for argument `factor` when it is 0, or when
    ///   an output dimension overflows `usize`.
    /// - `Error::UnsupportedDType` for a dtype the backend does not store.
    fn pixel_shuffle(
        &self,
        input: &Tensor<R>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<Tensor<R>>;

    /// Run [`pixel_shuffle`](Self::pixel_shuffle) and write the result into `out`.
    ///
    /// `out` has the output shape and the input's dtype. Elements of the
    /// parent buffer outside `out` keep their values.
    ///
    /// # Aliasing
    ///
    /// `out` must not share an element with `input`. Views of one storage
    /// pass when they are provably disjoint, by the rule of
    /// [`FusedConvOps::conv2d_fused_into`](super::FusedConvOps::conv2d_fused_into).
    ///
    /// # Errors
    ///
    /// Every error of [`pixel_shuffle`](Self::pixel_shuffle), plus:
    /// - `Error::ShapeMismatch` when `out` has the wrong shape.
    /// - `Error::DTypeMismatch` when `out` has another dtype.
    /// - `Error::InvalidArgument` for argument `out` when it repeats an
    ///   element or overlaps `input`.
    fn pixel_shuffle_into(
        &self,
        out: &Tensor<R>,
        input: &Tensor<R>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<()>;

    /// Rearrange `[N, C, H·r, W·r]` into `[N, C·r², H, W]`, matching
    /// `torch.nn.functional.pixel_unshuffle`. It is the exact inverse of
    /// [`pixel_shuffle`](Self::pixel_shuffle).
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` for argument `input` when the rank is not 4,
    ///   or when H or W is not divisible by `factor`.
    /// - `Error::InvalidArgument` for argument `factor` when it is 0, or when
    ///   the output channel count overflows `usize`.
    /// - `Error::UnsupportedDType` for a dtype the backend does not store.
    fn pixel_unshuffle(
        &self,
        input: &Tensor<R>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<Tensor<R>>;

    /// Run [`pixel_unshuffle`](Self::pixel_unshuffle) and write the result into `out`.
    ///
    /// `out` follows the rules of [`pixel_shuffle_into`](Self::pixel_shuffle_into).
    ///
    /// # Errors
    ///
    /// Every error of [`pixel_unshuffle`](Self::pixel_unshuffle), plus the
    /// `out` errors of [`pixel_shuffle_into`](Self::pixel_shuffle_into).
    fn pixel_unshuffle_into(
        &self,
        out: &Tensor<R>,
        input: &Tensor<R>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<()>;

    /// Nearest-neighbour upsampling over H and W, matching
    /// `torch.nn.functional.interpolate(mode="nearest")` on CPU.
    ///
    /// Per axis, the source index is `min(floor(f32(dst) * scale), in - 1)`
    /// in f32. `scale` comes from [`NearestSize`]. When the output size
    /// equals the input size the index is `dst`. When it equals twice the
    /// input size the index is `dst / 2`. These two cases match PyTorch's
    /// `nearest_idx`. Integer factors give the same result as
    /// `repeat_interleave` along H and W.
    ///
    /// This is `nearest`, not `nearest-exact`.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` for argument `input` when the rank is not 4,
    ///   or when H or W is 0.
    /// - `Error::InvalidArgument` for argument `size` when a scale factor is
    ///   not finite and positive, or an output size is 0 or overflows `usize`.
    /// - `Error::UnsupportedDType` for a dtype the backend does not store.
    fn upsample_nearest2d(
        &self,
        input: &Tensor<R>,
        size: NearestSize,
        layout: ConvLayout,
    ) -> Result<Tensor<R>>;

    /// Run [`upsample_nearest2d`](Self::upsample_nearest2d) and write the result into `out`.
    ///
    /// `out` follows the rules of [`pixel_shuffle_into`](Self::pixel_shuffle_into).
    ///
    /// # Errors
    ///
    /// Every error of [`upsample_nearest2d`](Self::upsample_nearest2d), plus
    /// the `out` errors of [`pixel_shuffle_into`](Self::pixel_shuffle_into).
    fn upsample_nearest2d_into(
        &self,
        out: &Tensor<R>,
        input: &Tensor<R>,
        size: NearestSize,
        layout: ConvLayout,
    ) -> Result<()>;
}
