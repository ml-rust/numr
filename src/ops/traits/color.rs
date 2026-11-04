//! Video frame colour conversion: 4:2:0 YCbCr planes to and from float RGB,
//! and packed RGB24 to and from float RGB.

use super::{ConvLayout, PackedPlane, YuvPlanes, YuvSpec};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

fn not_implemented<T>(feature: &'static str) -> Result<T> {
    Err(Error::NotImplemented { feature })
}

/// Colour conversion of one video frame.
///
/// # Frames and layouts
///
/// Every call converts one frame. Float RGB has shape `[1, 3, H, W]` for
/// `ConvLayout::Nchw` and `[1, H, W, 3]` for `ConvLayout::Nhwc`, channels in
/// R, G, B order. For a batch, call an `_into` method once per frame on
/// `out.narrow(0, n, 1)` of a batch tensor. [`YuvPlanes`] lists the plane
/// shapes and dtypes. Packed RGB24 is U8 `[H, W, 3]`.
///
/// Every input and every `out` can be any strided view, such as a surface
/// with a row pitch. No operand is copied to contiguous first.
///
/// # Packed words
///
/// A plane or an RGB24 frame can also pass as [`PackedPlane`] words: U32
/// `[rows, n]` holding the bytes of each row from the start of its first
/// word. The `YuvPlanes` `Packed` variants and the `packed_rgb24` methods
/// take this form. Results are the same bits as the U8 and U16 form on
/// every backend.
///
/// - CPU and CUDA read and write the words in place as U8 or U16 samples.
/// - WebGPU kernels address 32-bit words only, so WebGPU takes planes and
///   RGB24 in this form only. Each invocation owns whole words of every
///   plane it writes, and merges a partial last word of a row with the
///   bytes it keeps.
/// - A row pitch in bytes converts to a word shape with
///   [`PackedPlane::words_shape`], which refuses a pitch that is not a
///   multiple of 4.
///
/// # YCbCr to RGB
///
/// With `bits` the sample depth, `s = 2^(bits - 8)` and `M = 2^bits - 1`, the
/// samples normalise to `Y'` in `[0, 1]` and `Cb'`, `Cr'` in `[-0.5, 0.5]`:
///
/// | Range | `Y'` | `Cb'`, `Cr'` |
/// | --- | --- | --- |
/// | `Limited` | `(Y - 16·s) / (219·s)` | `(C - 128·s) / (224·s)` |
/// | `Full` | `Y / M` | `(C - 2^(bits-1)) / M` |
///
/// Then, with `Kr`, `Kb` from [`ColorMatrix`](super::ColorMatrix) and
/// `Kg = 1 - Kr - Kb`:
///
/// | Matrix | `Kr` | `Kb` |
/// | --- | --- | --- |
/// | `Bt601` | 0.299 | 0.114 |
/// | `Bt709` | 0.2126 | 0.0722 |
/// | `Fcc` | 0.30 | 0.11 |
/// | `Smpte240m` | 0.212 | 0.087 |
///
/// ```text
/// R = Y' + 2(1 - Kr)·Cr'
/// B = Y' + 2(1 - Kb)·Cb'
/// G = (Y' - Kr·R - Kb·B) / Kg
/// ```
///
/// Each of R, G and B is clamped to `[0, 1]`, so the output matches a
/// saturating RGB24 decode divided by 255 up to quantisation. Models train on
/// images in `[0, 1]`, and the clamp keeps footroom, headroom and
/// out-of-gamut codes out of that range.
///
/// # Chroma upsampling
///
/// Chroma upsamples to 4:4:4 by bilinear interpolation at the luma sample
/// positions. Out-of-frame chroma indices clamp to the edge.
///
/// - Vertical, both sitings: luma row `2k` takes chroma rows `k - 1` and `k`
///   with weights `1/4` and `3/4`. Luma row `2k + 1` takes rows `k` and
///   `k + 1` with weights `3/4` and `1/4`.
/// - Horizontal `Center`: the vertical rule along columns.
/// - Horizontal `Left`: luma column `2k` takes chroma column `k`. Luma column
///   `2k + 1` takes columns `k` and `k + 1` with weights `1/2` each.
///
/// The weights are multiples of `1/16`, so the interpolated chroma is exact.
///
/// # RGB to YCbCr
///
/// R, G and B are first clamped to `[0, 1]`, as `clamp(0, 1)` before an
/// 8-bit store does. Then per pixel:
///
/// ```text
/// Y'  = Kr·R + Kg·G + Kb·B
/// Cb' = (B - Y') / (2(1 - Kb))
/// Cr' = (R - Y') / (2(1 - Kr))
/// ```
///
/// Chroma downsamples by a filter centred on the chroma sample position:
///
/// - `Center`: the mean of the 2x2 block of luma pixels.
/// - `Left`: taps `[1, 2, 1] / 4` over luma columns `2j - 1`, `2j`, `2j + 1`,
///   with column `-1` clamped to 0, averaged over rows `2i` and `2i + 1`.
///
/// Each sample quantises as `round_ties_even(clamp(v·mul + off, lo, hi))`
/// with the inverse of the normalisation above and `lo`, `hi` the code range
/// of [`ColorRange`](super::ColorRange). P010 stores `value << 6`.
///
/// # Rounding
///
/// Every backend evaluates in F32 with one rounding per operation, in this
/// order, and no fused multiply-add. Divisions are correctly rounded, so
/// limited-range white decodes to exactly 1:
///
/// - Decode: `Y' = f32(Y - y_off) / y_mul` and
///   `C' = f32(S - 16·c_off) / (16·c_mul)`, with `S` the interpolated chroma
///   in sixteenths and `y_mul`, `c_mul` the divisors of the table above.
///   Then `R = Y' + cr_r·Cr'`, `B = Y' + cb_b·Cb'` and
///   `G = (Y' + g_cb·Cb') + g_cr·Cr'`, with `cr_r = 2(1 - Kr)`,
///   `cb_b = 2(1 - Kb)`, `g_cb = -2·Kb·(1 - Kb) / Kg` and
///   `g_cr = -2·Kr·(1 - Kr) / Kg`. The last two expand the G formula above.
/// - Encode: `Y' = (Kr·R + Kg·G) + Kb·B`, `Cb' = (B - Y') / cb_b` and
///   `Cr' = (R - Y') / cr_r`. The `Center` mean is
///   `((c00 + c01) + (c10 + c11))·0.25`. The `Left` filter is
///   `(h0 + h1)·0.125` with `h = (c[2j-1] + 2·c[2j]) + c[2j+1]` per row.
///
/// Each coefficient is computed in F64 and rounded to F32 once. Every
/// backend runs the same operations on the same constants, so integer and
/// F32 outputs match bit for bit across backends. The CUDA build flushes a
/// subnormal F32 input to zero, which moves no output code.
///
/// WGSL leaves the rounding direction of `+`, `-` and `*` to the
/// implementation, allows 2.5 ULP for `/`, can flush subnormals, and can
/// fuse or reassociate float operations. The WebGPU kernels therefore run
/// each F32 operation in integer arithmetic on the bit pattern, which WGSL
/// defines exactly, and round to nearest even with subnormals kept.
///
/// # Dtypes
///
/// | Backend | Float RGB |
/// | --- | --- |
/// | CPU | F32, F64, F16, BF16 |
/// | CUDA | F32, F64, F16, BF16 |
/// | WebGPU | F32 |
///
/// F16 and BF16 round once at the store, to nearest even. F64 holds the F32
/// result exactly, and an F64 input rounds to F32 at the load. CPU needs the
/// `f16` feature for F16 and BF16. A NaN input reads as 0 at the clamp.
///
/// # Aliasing
///
/// No `out` and no written plane can repeat an element. Written views must
/// share no byte with each other or with a read operand.
///
/// # Backends
///
/// CPU, CUDA and WebGPU implement every method. WebGPU returns
/// `Error::BackendLimitation` for U8 and U16 planes and U8 RGB24, and names
/// the packed form to pass instead. A backend without colour kernels takes
/// the default methods, which return `Error::NotImplemented`.
///
/// # Autograd
///
/// No gradient is recorded.
pub trait ColorOps<R: Runtime> {
    /// Convert 4:2:0 planes to float RGB in `out_dtype`.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` for argument `planes` when a plane has the
    ///   wrong rank, shape or dtype, or when H or W is odd. A packed plane
    ///   also fails with a column stride other than 1 or too few words per
    ///   row. The reason names the plane and the dimension.
    /// - `Error::UnsupportedDType` for an `out_dtype` the backend does not
    ///   produce.
    /// - `Error::BackendLimitation` on WebGPU for planes that are not packed.
    /// - `Error::NotImplemented` on a backend without colour kernels.
    fn yuv_to_rgb(
        &self,
        planes: YuvPlanes<'_, R>,
        spec: &YuvSpec,
        out_dtype: DType,
        layout: ConvLayout,
    ) -> Result<Tensor<R>> {
        let _ = (planes, spec, out_dtype, layout);
        not_implemented("ColorOps::yuv_to_rgb")
    }

    /// Run [`yuv_to_rgb`](Self::yuv_to_rgb) and write the result into `out`.
    ///
    /// `out` has the RGB shape of `layout` and a float dtype. Elements of the
    /// parent buffer outside `out` keep their values.
    ///
    /// # Errors
    ///
    /// Every error of [`yuv_to_rgb`](Self::yuv_to_rgb), plus:
    /// - `Error::ShapeMismatch` when `out` has the wrong shape.
    /// - `Error::InvalidArgument` for argument `out` when it repeats an
    ///   element or overlaps a plane.
    fn yuv_to_rgb_into(
        &self,
        out: &Tensor<R>,
        planes: YuvPlanes<'_, R>,
        spec: &YuvSpec,
        layout: ConvLayout,
    ) -> Result<()> {
        let _ = (out, planes, spec, layout);
        not_implemented("ColorOps::yuv_to_rgb_into")
    }

    /// Convert float RGB to 4:2:0 and write the caller's `planes`.
    ///
    /// `rgb` has the RGB shape of `layout`. Elements of each plane's parent
    /// buffer outside the plane view keep their values.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` for argument `planes` as in
    ///   [`yuv_to_rgb`](Self::yuv_to_rgb), or when a plane repeats an element
    ///   or overlaps another plane or `rgb`.
    /// - `Error::ShapeMismatch` when `rgb` does not match the planes' frame.
    /// - `Error::UnsupportedDType` for an `rgb` dtype the backend does not
    ///   read.
    /// - `Error::BackendLimitation` on WebGPU for planes that are not packed.
    /// - `Error::NotImplemented` on a backend without colour kernels.
    fn rgb_to_yuv_into(
        &self,
        rgb: &Tensor<R>,
        planes: YuvPlanes<'_, R>,
        spec: &YuvSpec,
        layout: ConvLayout,
    ) -> Result<()> {
        let _ = (rgb, planes, spec, layout);
        not_implemented("ColorOps::rgb_to_yuv_into")
    }

    /// Unpack U8 `[H, W, 3]` RGB24 into float RGB: `f32(v) / 255` in F32,
    /// one correctly rounded division.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` for argument `rgb24` when it is not U8
    ///   `[H, W, 3]`.
    /// - `Error::UnsupportedDType` for an `out_dtype` the backend does not
    ///   produce.
    /// - `Error::BackendLimitation` on WebGPU, which takes
    ///   [`packed_rgb24_to_float`](Self::packed_rgb24_to_float) instead.
    /// - `Error::NotImplemented` on a backend without colour kernels.
    fn rgb24_to_float(
        &self,
        rgb24: &Tensor<R>,
        out_dtype: DType,
        layout: ConvLayout,
    ) -> Result<Tensor<R>> {
        let _ = (rgb24, out_dtype, layout);
        not_implemented("ColorOps::rgb24_to_float")
    }

    /// Run [`rgb24_to_float`](Self::rgb24_to_float) and write the result
    /// into `out`.
    ///
    /// # Errors
    ///
    /// Every error of [`rgb24_to_float`](Self::rgb24_to_float), plus the
    /// `out` errors of [`yuv_to_rgb_into`](Self::yuv_to_rgb_into).
    fn rgb24_to_float_into(
        &self,
        out: &Tensor<R>,
        rgb24: &Tensor<R>,
        layout: ConvLayout,
    ) -> Result<()> {
        let _ = (out, rgb24, layout);
        not_implemented("ColorOps::rgb24_to_float_into")
    }

    /// Pack float RGB into a new U8 `[H, W, 3]` RGB24 tensor.
    ///
    /// Each value is `round_ties_even(clamp(x, 0, 1)·255)` in F32, matching
    /// `x.clamp(0, 1).mul(255).round().to(torch.uint8)` on an F32 tensor.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` for argument `rgb` when it does not have
    ///   the RGB shape of `layout`.
    /// - `Error::UnsupportedDType` for an `rgb` dtype the backend does not
    ///   read.
    /// - `Error::BackendLimitation` on WebGPU, which takes
    ///   [`float_to_packed_rgb24`](Self::float_to_packed_rgb24) instead.
    /// - `Error::NotImplemented` on a backend without colour kernels.
    fn float_to_rgb24(&self, rgb: &Tensor<R>, layout: ConvLayout) -> Result<Tensor<R>> {
        let _ = (rgb, layout);
        not_implemented("ColorOps::float_to_rgb24")
    }

    /// Run [`float_to_rgb24`](Self::float_to_rgb24) and write the result
    /// into the U8 `[H, W, 3]` view `out`.
    ///
    /// # Errors
    ///
    /// Every error of [`float_to_rgb24`](Self::float_to_rgb24), plus:
    /// - `Error::ShapeMismatch` when `out` has the wrong shape.
    /// - `Error::InvalidArgument` for argument `out` when it is not U8,
    ///   repeats an element or overlaps `rgb`.
    fn float_to_rgb24_into(
        &self,
        out: &Tensor<R>,
        rgb: &Tensor<R>,
        layout: ConvLayout,
    ) -> Result<()> {
        let _ = (out, rgb, layout);
        not_implemented("ColorOps::float_to_rgb24_into")
    }

    /// Unpack packed RGB24 words into float RGB, as
    /// [`rgb24_to_float`](Self::rgb24_to_float).
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` for argument `rgb24` when its words are not
    ///   U32 `[H, n]` with column stride 1 and `n >= ceil(3·W / 4)`.
    /// - `Error::UnsupportedDType` for an `out_dtype` the backend does not
    ///   produce.
    /// - `Error::NotImplemented` on a backend without colour kernels.
    fn packed_rgb24_to_float(
        &self,
        rgb24: PackedPlane<'_, R>,
        out_dtype: DType,
        layout: ConvLayout,
    ) -> Result<Tensor<R>> {
        let _ = (rgb24, out_dtype, layout);
        not_implemented("ColorOps::packed_rgb24_to_float")
    }

    /// Run [`packed_rgb24_to_float`](Self::packed_rgb24_to_float) and write
    /// the result into `out`.
    ///
    /// # Errors
    ///
    /// Every error of [`packed_rgb24_to_float`](Self::packed_rgb24_to_float),
    /// plus the `out` errors of [`yuv_to_rgb_into`](Self::yuv_to_rgb_into).
    fn packed_rgb24_to_float_into(
        &self,
        out: &Tensor<R>,
        rgb24: PackedPlane<'_, R>,
        layout: ConvLayout,
    ) -> Result<()> {
        let _ = (out, rgb24, layout);
        not_implemented("ColorOps::packed_rgb24_to_float_into")
    }

    /// Pack float RGB into new RGB24 words, U32 `[H, ceil(3·W / 4)]`, as
    /// [`float_to_rgb24`](Self::float_to_rgb24). Bytes past `3·W` in the
    /// last word of a row are 0. Pass the result as `PackedPlane::new(&t, W)`.
    ///
    /// # Errors
    ///
    /// Every error of [`float_to_rgb24`](Self::float_to_rgb24) but the
    /// WebGPU one.
    fn float_to_packed_rgb24(&self, rgb: &Tensor<R>, layout: ConvLayout) -> Result<Tensor<R>> {
        let _ = (rgb, layout);
        not_implemented("ColorOps::float_to_packed_rgb24")
    }

    /// Run [`float_to_packed_rgb24`](Self::float_to_packed_rgb24) and write
    /// the result into the packed words `out`. Bytes of a row past `3·W`
    /// keep their values.
    ///
    /// # Errors
    ///
    /// Every error of [`float_to_packed_rgb24`](Self::float_to_packed_rgb24),
    /// plus:
    /// - `Error::InvalidArgument` for argument `out` when its words are not
    ///   U32 `[H, n]` with column stride 1 and `n >= ceil(3·W / 4)`, repeat a
    ///   word or overlap `rgb`.
    /// - `Error::ShapeMismatch` when `out` holds another `[H, W]`.
    fn float_to_packed_rgb24_into(
        &self,
        out: PackedPlane<'_, R>,
        rgb: &Tensor<R>,
        layout: ConvLayout,
    ) -> Result<()> {
        let _ = (out, rgb, layout);
        not_implemented("ColorOps::float_to_packed_rgb24_into")
    }

    /// Replicate-pad a packed luma plane on the bottom and right into new
    /// words, U32 `[rows, ceil(columns · sample_bytes / 4)]`.
    ///
    /// The YCbCr methods refuse an odd H or W. Pad the luma of a frame with an
    /// odd display size to the next even size first. Its chroma has that even
    /// size's shape already, with `ceil(W / 2)` samples per row.
    ///
    /// `sample` is `DType::U8` for NV12 and I420 or `DType::U16` for P010,
    /// because packed words do not record their sample size. Sample
    /// `[r, c]` of the result is sample `[min(r, H - 1), min(c, W - 1)]` of
    /// `y`, every stored bit included. Bytes past `columns` in the last word
    /// of a row are 0. Bytes of `y` past its width are not read. Pass the
    /// result as `PackedPlane::new(&t, columns)`. Target sizes need not be
    /// even, and equal sizes give a tight copy.
    ///
    /// # Errors
    ///
    /// - `Error::InvalidArgument` for argument `sample` when it is neither
    ///   U8 nor U16.
    /// - `Error::InvalidArgument` for argument `y` when its words are not U32
    ///   `[H, n]` with column stride 1 and enough words per row, or `H` or
    ///   `W` is 0.
    /// - `Error::InvalidArgument` for argument `rows` or `columns` when the
    ///   target is below the source size.
    /// - `Error::NotImplemented` on a backend without colour kernels.
    fn pad_packed_luma(
        &self,
        y: PackedPlane<'_, R>,
        rows: usize,
        columns: usize,
        sample: DType,
    ) -> Result<Tensor<R>> {
        let _ = (y, rows, columns, sample);
        not_implemented("ColorOps::pad_packed_luma")
    }
}
