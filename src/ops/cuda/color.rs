//! CUDA implementation of `ColorOps`.
//!
//! Plane views, RGB views and RGB24 views pass to the kernels with their
//! strides, so no operand is copied to contiguous first and no data leaves
//! the device.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::common::{
    YuvFrame, packed_rgb24_shape, plan_decode, plan_decode_into, plan_encode, plan_pack,
    plan_pack_into, plan_pack_packed_into, plan_pad_packed_luma, plan_unpack, plan_unpack_into,
    plan_unpack_packed, plan_unpack_packed_into, rgb_shape, rgb_strides, rgb24_strides,
};
use crate::ops::{ColorOps, ConvLayout, PackedPlane, YuvPlanes, YuvSpec};
use crate::runtime::cuda::kernels::{
    Rgb24Direction, Rgb24Launch, YuvDirection, launch_rgb24, launch_yuv,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Float dtypes the CUDA colour kernels read and write.
fn float_ok(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16)
}

impl CudaClient {
    /// Run a YCbCr kernel between `frame` and the RGB view `rgb`, which
    /// passed the checks.
    fn color_yuv_run(
        &self,
        direction: YuvDirection,
        frame: &YuvFrame,
        rgb: &Tensor<CudaRuntime>,
        layout: ConvLayout,
    ) -> Result<()> {
        // SAFETY: `plan_yuv_frame` built `frame` from device views, `rgb`
        // has the frame's RGB shape, `ptr()` folds in its offset, and the
        // written views passed the repeat and overlap checks.
        unsafe {
            launch_yuv(
                &self.context,
                &self.stream,
                self.device.index,
                direction,
                frame,
                rgb.dtype(),
                rgb.ptr(),
                rgb_strides(rgb.strides(), layout),
            )
        }
    }

    /// Run an RGB24 kernel between the RGB24 view `rgb24` and the float RGB
    /// view `rgb`, which passed the checks.
    fn color_rgb24_run(
        &self,
        direction: Rgb24Direction,
        rgb24: &Tensor<CudaRuntime>,
        rgb: &Tensor<CudaRuntime>,
        layout: ConvLayout,
        size: [usize; 2],
    ) -> Result<()> {
        let packed = (rgb24.ptr(), rgb24_strides(rgb24.strides()));
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
        // SAFETY: both views have the checked shapes, `ptr()` folds in each
        // offset, and the destination passed the repeat and overlap checks.
        unsafe { launch_rgb24(&self.context, &self.stream, self.device.index, &launch) }
    }
}

/// ColorOps implementation for CUDA runtime. F32, F64, F16 and BF16. Packed
/// planes and packed RGB24 are read and written in place as bytes.
impl ColorOps<CudaRuntime> for CudaClient {
    fn yuv_to_rgb(
        &self,
        planes: YuvPlanes<'_, CudaRuntime>,
        spec: &YuvSpec,
        out_dtype: DType,
        layout: ConvLayout,
    ) -> Result<Tensor<CudaRuntime>> {
        let frame = plan_decode(&planes, spec, out_dtype, float_ok, "yuv_to_rgb")?;
        let shape = rgb_shape(layout, frame.height, frame.width);
        let out = Tensor::<CudaRuntime>::empty(&shape, out_dtype, &self.device)?;
        self.color_yuv_run(YuvDirection::ToRgb, &frame, &out, layout)?;
        Ok(out)
    }

    fn yuv_to_rgb_into(
        &self,
        out: &Tensor<CudaRuntime>,
        planes: YuvPlanes<'_, CudaRuntime>,
        spec: &YuvSpec,
        layout: ConvLayout,
    ) -> Result<()> {
        let op = "yuv_to_rgb_into";
        let frame = plan_decode_into(out, &planes, spec, layout, float_ok, op)?;
        self.color_yuv_run(YuvDirection::ToRgb, &frame, out, layout)
    }

    fn rgb_to_yuv_into(
        &self,
        rgb: &Tensor<CudaRuntime>,
        planes: YuvPlanes<'_, CudaRuntime>,
        spec: &YuvSpec,
        layout: ConvLayout,
    ) -> Result<()> {
        let frame = plan_encode(rgb, &planes, spec, layout, float_ok, "rgb_to_yuv_into")?;
        self.color_yuv_run(YuvDirection::ToYuv, &frame, rgb, layout)
    }

    fn rgb24_to_float(
        &self,
        rgb24: &Tensor<CudaRuntime>,
        out_dtype: DType,
        layout: ConvLayout,
    ) -> Result<Tensor<CudaRuntime>> {
        let [h, w] = plan_unpack(rgb24, out_dtype, float_ok, "rgb24_to_float")?;
        let out = Tensor::<CudaRuntime>::empty(&rgb_shape(layout, h, w), out_dtype, &self.device)?;
        self.color_rgb24_run(Rgb24Direction::Unpack, rgb24, &out, layout, [h, w])?;
        Ok(out)
    }

    fn rgb24_to_float_into(
        &self,
        out: &Tensor<CudaRuntime>,
        rgb24: &Tensor<CudaRuntime>,
        layout: ConvLayout,
    ) -> Result<()> {
        let size = plan_unpack_into(out, rgb24, layout, float_ok, "rgb24_to_float_into")?;
        self.color_rgb24_run(Rgb24Direction::Unpack, rgb24, out, layout, size)
    }

    fn float_to_rgb24(
        &self,
        rgb: &Tensor<CudaRuntime>,
        layout: ConvLayout,
    ) -> Result<Tensor<CudaRuntime>> {
        let [h, w] = plan_pack(rgb, layout, float_ok, "float_to_rgb24")?;
        let out = Tensor::<CudaRuntime>::empty(&[h, w, 3], DType::U8, &self.device)?;
        self.color_rgb24_run(Rgb24Direction::Pack, &out, rgb, layout, [h, w])?;
        Ok(out)
    }

    fn float_to_rgb24_into(
        &self,
        out: &Tensor<CudaRuntime>,
        rgb: &Tensor<CudaRuntime>,
        layout: ConvLayout,
    ) -> Result<()> {
        let size = plan_pack_into(out, rgb, layout, float_ok, "float_to_rgb24_into")?;
        self.color_rgb24_run(Rgb24Direction::Pack, out, rgb, layout, size)
    }

    fn packed_rgb24_to_float(
        &self,
        rgb24: PackedPlane<'_, CudaRuntime>,
        out_dtype: DType,
        layout: ConvLayout,
    ) -> Result<Tensor<CudaRuntime>> {
        let op = "packed_rgb24_to_float";
        let [h, w] = plan_unpack_packed(&rgb24, out_dtype, float_ok, op)?;
        let out = Tensor::<CudaRuntime>::empty(&rgb_shape(layout, h, w), out_dtype, &self.device)?;
        self.color_packed_rgb24_run(Rgb24Direction::Unpack, &rgb24, &out, layout, [h, w])?;
        Ok(out)
    }

    fn packed_rgb24_to_float_into(
        &self,
        out: &Tensor<CudaRuntime>,
        rgb24: PackedPlane<'_, CudaRuntime>,
        layout: ConvLayout,
    ) -> Result<()> {
        let op = "packed_rgb24_to_float_into";
        let size = plan_unpack_packed_into(out, &rgb24, layout, float_ok, op)?;
        self.color_packed_rgb24_run(Rgb24Direction::Unpack, &rgb24, out, layout, size)
    }

    fn float_to_packed_rgb24(
        &self,
        rgb: &Tensor<CudaRuntime>,
        layout: ConvLayout,
    ) -> Result<Tensor<CudaRuntime>> {
        let size = plan_pack(rgb, layout, float_ok, "float_to_packed_rgb24")?;
        let words =
            Tensor::<CudaRuntime>::zeros(&packed_rgb24_shape(size), DType::U32, &self.device)?;
        let out = PackedPlane::new(&words, size[1]);
        self.color_packed_rgb24_run(Rgb24Direction::Pack, &out, rgb, layout, size)?;
        Ok(words)
    }

    fn float_to_packed_rgb24_into(
        &self,
        out: PackedPlane<'_, CudaRuntime>,
        rgb: &Tensor<CudaRuntime>,
        layout: ConvLayout,
    ) -> Result<()> {
        let op = "float_to_packed_rgb24_into";
        let size = plan_pack_packed_into(&out, rgb, layout, float_ok, op)?;
        self.color_packed_rgb24_run(Rgb24Direction::Pack, &out, rgb, layout, size)
    }

    fn pad_packed_luma(
        &self,
        y: PackedPlane<'_, CudaRuntime>,
        rows: usize,
        columns: usize,
        sample: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        let plan = plan_pad_packed_luma(&y, rows, columns, sample, "pad_packed_luma")?;
        let out = Tensor::<CudaRuntime>::zeros(&plan.out_shape(), DType::U32, &self.device)?;
        self.color_pad_packed_luma_run(&y, &plan, &out)?;
        Ok(out)
    }
}
