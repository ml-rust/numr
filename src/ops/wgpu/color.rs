//! WebGPU `ColorOps` on packed U32 words and F32 RGB.
//!
//! WGSL kernels address 32-bit words only, so planes and RGB24 frames pass
//! as `PackedPlane` words. The U8 and U16 forms return
//! `Error::BackendLimitation` that names the packed form.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::common::{
    packed_rgb24_shape, plan_decode, plan_decode_into, plan_encode, plan_pack,
    plan_pack_packed_into, plan_pad_packed_luma, plan_unpack_packed, plan_unpack_packed_into,
    rgb_shape,
};
use crate::ops::{ColorOps, ConvLayout, PackedPlane, YuvPlanes, YuvSpec};
use crate::runtime::RuntimeClient;
use crate::runtime::wgpu::ops::native::{
    native_pad_packed_luma, native_rgb24, native_yuv, packed_rgb24_only, require_packed,
};
use crate::runtime::wgpu::shaders::color::ColorEntry;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// The float dtype the WebGPU colour kernels read and write.
fn float_ok(dtype: DType) -> bool {
    dtype == DType::F32
}

/// ColorOps implementation for WebGPU runtime. F32 RGB, packed planes and
/// packed RGB24.
impl ColorOps<WgpuRuntime> for WgpuClient {
    fn yuv_to_rgb(
        &self,
        planes: YuvPlanes<'_, WgpuRuntime>,
        spec: &YuvSpec,
        out_dtype: DType,
        layout: ConvLayout,
    ) -> Result<Tensor<WgpuRuntime>> {
        let op = "yuv_to_rgb";
        require_packed(&planes, op)?;
        let frame = plan_decode(&planes, spec, out_dtype, float_ok, op)?;
        let shape = rgb_shape(layout, frame.height, frame.width);
        let out = Tensor::<WgpuRuntime>::empty(&shape, out_dtype, self.device())?;
        native_yuv(self, ColorEntry::YuvToRgb, &frame, &planes, &out, layout)?;
        Ok(out)
    }

    fn yuv_to_rgb_into(
        &self,
        out: &Tensor<WgpuRuntime>,
        planes: YuvPlanes<'_, WgpuRuntime>,
        spec: &YuvSpec,
        layout: ConvLayout,
    ) -> Result<()> {
        let op = "yuv_to_rgb_into";
        require_packed(&planes, op)?;
        let frame = plan_decode_into(out, &planes, spec, layout, float_ok, op)?;
        native_yuv(self, ColorEntry::YuvToRgb, &frame, &planes, out, layout)
    }

    fn rgb_to_yuv_into(
        &self,
        rgb: &Tensor<WgpuRuntime>,
        planes: YuvPlanes<'_, WgpuRuntime>,
        spec: &YuvSpec,
        layout: ConvLayout,
    ) -> Result<()> {
        let op = "rgb_to_yuv_into";
        require_packed(&planes, op)?;
        let frame = plan_encode(rgb, &planes, spec, layout, float_ok, op)?;
        native_yuv(self, ColorEntry::RgbToYuv, &frame, &planes, rgb, layout)
    }

    fn rgb24_to_float(
        &self,
        _rgb24: &Tensor<WgpuRuntime>,
        _out_dtype: DType,
        _layout: ConvLayout,
    ) -> Result<Tensor<WgpuRuntime>> {
        Err(packed_rgb24_only("rgb24_to_float", "packed_rgb24_to_float"))
    }

    fn rgb24_to_float_into(
        &self,
        _out: &Tensor<WgpuRuntime>,
        _rgb24: &Tensor<WgpuRuntime>,
        _layout: ConvLayout,
    ) -> Result<()> {
        Err(packed_rgb24_only(
            "rgb24_to_float_into",
            "packed_rgb24_to_float_into",
        ))
    }

    fn float_to_rgb24(
        &self,
        _rgb: &Tensor<WgpuRuntime>,
        _layout: ConvLayout,
    ) -> Result<Tensor<WgpuRuntime>> {
        Err(packed_rgb24_only("float_to_rgb24", "float_to_packed_rgb24"))
    }

    fn float_to_rgb24_into(
        &self,
        _out: &Tensor<WgpuRuntime>,
        _rgb: &Tensor<WgpuRuntime>,
        _layout: ConvLayout,
    ) -> Result<()> {
        Err(packed_rgb24_only(
            "float_to_rgb24_into",
            "float_to_packed_rgb24_into",
        ))
    }

    fn packed_rgb24_to_float(
        &self,
        rgb24: PackedPlane<'_, WgpuRuntime>,
        out_dtype: DType,
        layout: ConvLayout,
    ) -> Result<Tensor<WgpuRuntime>> {
        let op = "packed_rgb24_to_float";
        let size = plan_unpack_packed(&rgb24, out_dtype, float_ok, op)?;
        let shape = rgb_shape(layout, size[0], size[1]);
        let out = Tensor::<WgpuRuntime>::empty(&shape, out_dtype, self.device())?;
        native_rgb24(
            self,
            ColorEntry::Rgb24ToFloat,
            &rgb24,
            &out,
            layout,
            size,
            false,
        )?;
        Ok(out)
    }

    fn packed_rgb24_to_float_into(
        &self,
        out: &Tensor<WgpuRuntime>,
        rgb24: PackedPlane<'_, WgpuRuntime>,
        layout: ConvLayout,
    ) -> Result<()> {
        let op = "packed_rgb24_to_float_into";
        let size = plan_unpack_packed_into(out, &rgb24, layout, float_ok, op)?;
        native_rgb24(
            self,
            ColorEntry::Rgb24ToFloat,
            &rgb24,
            out,
            layout,
            size,
            false,
        )
    }

    fn float_to_packed_rgb24(
        &self,
        rgb: &Tensor<WgpuRuntime>,
        layout: ConvLayout,
    ) -> Result<Tensor<WgpuRuntime>> {
        let size = plan_pack(rgb, layout, float_ok, "float_to_packed_rgb24")?;
        let words =
            Tensor::<WgpuRuntime>::empty(&packed_rgb24_shape(size), DType::U32, self.device())?;
        let out = PackedPlane::new(&words, size[1]);
        native_rgb24(
            self,
            ColorEntry::FloatToRgb24,
            &out,
            rgb,
            layout,
            size,
            true,
        )?;
        Ok(words)
    }

    fn float_to_packed_rgb24_into(
        &self,
        out: PackedPlane<'_, WgpuRuntime>,
        rgb: &Tensor<WgpuRuntime>,
        layout: ConvLayout,
    ) -> Result<()> {
        let op = "float_to_packed_rgb24_into";
        let size = plan_pack_packed_into(&out, rgb, layout, float_ok, op)?;
        native_rgb24(
            self,
            ColorEntry::FloatToRgb24,
            &out,
            rgb,
            layout,
            size,
            false,
        )
    }

    fn pad_packed_luma(
        &self,
        y: PackedPlane<'_, WgpuRuntime>,
        rows: usize,
        columns: usize,
        sample: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        let plan = plan_pad_packed_luma(&y, rows, columns, sample, "pad_packed_luma")?;
        let out = Tensor::<WgpuRuntime>::empty(&plan.out_shape(), DType::U32, self.device())?;
        native_pad_packed_luma(self, &y, &plan, &out)?;
        Ok(out)
    }
}
