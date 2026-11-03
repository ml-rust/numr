//! CUDA launch of the packed RGB24 `ColorOps` methods.
//!
//! Packed words are the bytes of the frame, so the RGB24 kernels of
//! `color.cu` read and write them in place through byte strides.

use crate::error::Result;
use crate::ops::common::{packed_rgb24_strides, rgb_strides};
use crate::ops::{ConvLayout, PackedPlane};
use crate::runtime::cuda::kernels::{Rgb24Direction, Rgb24Launch, launch_rgb24};
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
}
