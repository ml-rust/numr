//! CUDA launch of the packed RGB24 `ColorOps` methods and packed luma padding.
//!
//! Packed words are the bytes of the frame, so the RGB24 kernels of
//! `color.cu` read and write them in place through byte strides. Padding runs
//! the strided copy kernel on the samples of the words.

use crate::error::Result;
use crate::ops::common::color_packed::packed_view;
use crate::ops::common::{PackedLumaPad, packed_rgb24_strides, rgb_strides};
use crate::ops::{ConvLayout, PackedPlane};
use crate::runtime::cuda::kernels::{
    Rgb24Direction, Rgb24Launch, launch_rgb24, launch_strided_copy_into,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl CudaClient {
    /// Run an RGB24 kernel between the checked packed words `rgb24` and the
    /// float RGB view `rgb`.
    pub(super) fn color_packed_rgb24_run(
        &self,
        direction: Rgb24Direction,
        rgb24: &PackedPlane<'_, CudaRuntime>,
        rgb: &Tensor<CudaRuntime>,
        layout: ConvLayout,
        size: [usize; 2],
    ) -> Result<()> {
        let (arg, op) = match direction {
            Rgb24Direction::Unpack => ("rgb24", "packed_rgb24_to_float"),
            Rgb24Direction::Pack => ("out", "float_to_packed_rgb24"),
        };
        let packed = (rgb24.words.ptr(), packed_rgb24_strides(rgb24, arg, op)?);
        let float = (rgb.ptr(), rgb_strides(rgb.strides(), layout));
        let ((src, src_strides), (dst, dst_strides)) = match direction {
            Rgb24Direction::Unpack => (packed, float),
            Rgb24Direction::Pack => (float, packed),
        };
        let launch = Rgb24Launch {
            direction,
            float: rgb.dtype(),
            size,
            src,
            dst,
            src_strides,
            dst_strides,
        };
        // SAFETY: the words hold `3·W` bytes per row from the first word of
        // each row, `ptr()` folds in each offset, and the destination passed
        // the repeat and overlap checks.
        unsafe { launch_rgb24(&self.context, &self.stream, self.device.index, &launch) }
    }

    /// Replicate-pad the checked packed luma `y` into the zeroed words `out`,
    /// which have the shape of `plan`.
    ///
    /// Three copies run in stream order: the source samples, the last source
    /// column into the right pad, and the last row of `out` into the bottom
    /// pad. A stride of 0 repeats the edge. An empty pad launches nothing.
    pub(super) fn color_pad_packed_luma_run(
        &self,
        y: &PackedPlane<'_, CudaRuntime>,
        plan: &PackedLumaPad,
        out: &Tensor<CudaRuntime>,
    ) -> Result<()> {
        let view = packed_view(y, plan.sample_bytes, "pad_packed_luma")?;
        let (src, src_rs) = (view.addr, view.row_stride);
        let sample = plan.sample_bytes;
        let [h, w] = plan.source;
        let [rows, columns] = plan.target;
        let dst = out.ptr();
        let dst_rs = (plan.row_words * 4 / sample) as isize;
        let copy = |from, shape: [usize; 2], from_strides: [isize; 2], from_at, to_at| {
            // SAFETY: the check found `W` samples in each of the `H` rows of
            // `y`, and `out` has `rows` rows of `columns` samples. `out` is
            // new, so a read of `y` shares no byte with it. The bottom pad
            // reads row `H - 1` of `out` and writes rows from `H`.
            unsafe {
                launch_strided_copy_into(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    from,
                    dst,
                    &shape,
                    &from_strides,
                    &[dst_rs, 1],
                    sample,
                    from_at,
                    to_at,
                )
            }
        };
        copy(src, [h, w], [src_rs, 1], 0, 0)?;
        copy(
            src,
            [h, columns - w],
            [src_rs, 0],
            (w - 1) * sample,
            w * sample,
        )?;
        let row_bytes = plan.row_words * 4;
        copy(
            dst,
            [rows - h, columns],
            [0, 1],
            (h - 1) * row_bytes,
            h * row_bytes,
        )
    }
}
