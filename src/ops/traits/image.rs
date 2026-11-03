//! Image operations: pixel shuffle, pixel unshuffle, nearest-neighbour
//! upsampling and bilinear or bicubic resize.
//!
//! The data-movement ops compute nothing, so every backend covers every dtype
//! it stores. `resize` interpolates and covers the float dtypes listed on it.
//! The `_into` variants write into a caller-given view, such as a channel
//! slice of a larger buffer.

use super::{ConvLayout, ResizeConfig};
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
/// For the data-movement ops, CPU and CUDA support every dtype and WebGPU
/// supports F32, I32 and U32. [`resize`](ImageOps::resize) lists its own.
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

    /// Resize H and W to `config.size`, matching PyTorch CPU
    /// `F.interpolate(size=..., mode="bilinear" | "bicubic",
    /// align_corners=False, antialias=...)`.
    ///
    /// `align_corners` is always false, as `F.interpolate` with `size=` given.
    /// Per axis `scale = in / out`, divided in the accumulation type. The
    /// output is not clamped, so bicubic overshoot stays.
    ///
    /// # Passes
    ///
    /// The resize runs in two separable passes: W first into an intermediate
    /// buffer in the accumulation type, then H into the output. This is
    /// PyTorch's CPU order. With antialias, an axis whose size does not
    /// change is skipped, and equal sizes on both axes copy the input.
    /// Without antialias both passes always run, as PyTorch's do: equal sizes
    /// return the input for finite values, and an `inf` or NaN reaches its
    /// neighbours through zero weights.
    ///
    /// # Taps and weights
    ///
    /// Each output index `i` computes its own window and weights:
    ///
    /// - Bilinear: `src = max(scale·(i + 0.5) − 0.5, 0)`. The taps are
    ///   `floor(src)` and the next index, clamped to the last pixel, with
    ///   weights `1 − λ` and `λ`. An axis of equal sizes uses pixel `i` with
    ///   weights 1 and 0.
    /// - Bicubic: `src = scale·(i + 0.5) − 0.5` without the clamp. The 4 taps
    ///   start at `floor(src) − 1`, each clamped to the border, with
    ///   `a = −0.75`.
    /// - Antialias: `support = (1 | 2)·max(scale, 1)` and
    ///   `center = scale·(i + 0.5)`. The taps run from
    ///   `max(trunc(center − support + 0.5), 0)` up to
    ///   `min(trunc(center + support + 0.5), in)`. Input index `k` gets
    ///   `filter((k − center + 0.5) / max(scale, 1))`, divided by the sum of
    ///   the window. The filter is the triangle `1 − |x|` or the Keys cubic
    ///   with `a = −0.5`.
    ///
    /// # Rounding
    ///
    /// Each output sums its `n` taps in order, in the accumulation type. The
    /// fused multiply-adds follow PyTorch's x86-64 AVX2 CPU kernel:
    ///
    /// - Without antialias, `acc = fma(x0, w0, x1·w1)`, then
    ///   `acc = fma(xj, wj, acc)`. The bicubic source index is
    ///   `fma(scale, i + 0.5, −0.5)`. The bilinear index and the bicubic
    ///   coefficients round every step.
    /// - With antialias, `acc = x0·w0`. Taps 1 to `B·floor((n − 1) / B)` add
    ///   rounded products, and later taps fuse. `B` is 4 for F32 accumulation
    ///   and 2 for F64. The cubic filter polynomial fuses.
    ///
    /// Other PyTorch builds can differ in the last bits.
    ///
    /// # Dtypes
    ///
    /// | Backend | Dtypes | Accumulation |
    /// | --- | --- | --- |
    /// | CPU, CUDA | F32, F64, F16, BF16 | F32, or F64 for F64 |
    /// | WebGPU | F32 | F32 |
    ///
    /// F16 and BF16 load into F32 and round once at the store. CPU needs the
    /// `f16` feature for them. CUDA matches CPU bit for bit, except that the
    /// CUDA build flushes F32 subnormals to zero. WebGPU matches CPU within
    /// rounding.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` for argument `input` when the rank is not 4,
    ///   or when H or W is 0.
    /// - `Error::InvalidArgument` for argument `config` when an output size
    ///   is 0, or when the output or the `[N, C, H_in, W_out]` intermediate
    ///   holds more than `isize::MAX` elements.
    /// - `Error::UnsupportedDType` for a dtype the backend does not resize.
    fn resize(
        &self,
        input: &Tensor<R>,
        config: &ResizeConfig,
        layout: ConvLayout,
    ) -> Result<Tensor<R>>;

    /// Run [`resize`](Self::resize) and write the result into `out`.
    ///
    /// `out` follows the rules of [`pixel_shuffle_into`](Self::pixel_shuffle_into).
    ///
    /// # Errors
    ///
    /// Every error of [`resize`](Self::resize), plus the `out` errors of
    /// [`pixel_shuffle_into`](Self::pixel_shuffle_into).
    fn resize_into(
        &self,
        out: &Tensor<R>,
        input: &Tensor<R>,
        config: &ResizeConfig,
        layout: ConvLayout,
    ) -> Result<()>;
}
