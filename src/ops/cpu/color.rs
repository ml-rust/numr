//! CPU implementation of `ColorOps`.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::common::{
    plan_decode, plan_decode_into, plan_encode, plan_pack, plan_pack_into, plan_unpack,
    plan_unpack_into, rgb_shape,
};
use crate::ops::{ColorOps, ConvLayout, YuvPlanes, YuvSpec};
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{
        color_float_ok, float_to_rgb24_run, rgb_to_yuv_run, rgb24_to_float_run, yuv_to_rgb_run,
    },
};
use crate::tensor::Tensor;

/// ColorOps implementation for CPU runtime. F32 and F64, plus F16 and BF16
/// with the `f16` feature.
impl ColorOps<CpuRuntime> for CpuClient {
    fn yuv_to_rgb(
        &self,
        planes: YuvPlanes<'_, CpuRuntime>,
        spec: &YuvSpec,
        out_dtype: DType,
        layout: ConvLayout,
    ) -> Result<Tensor<CpuRuntime>> {
        let frame = plan_decode(&planes, spec, out_dtype, color_float_ok, "yuv_to_rgb")?;
        let shape = rgb_shape(layout, frame.height, frame.width);
        let out = Tensor::<CpuRuntime>::empty(&shape, out_dtype, &self.device)?;
        yuv_to_rgb_run(self, &frame, &out, layout)?;
        Ok(out)
    }

    fn yuv_to_rgb_into(
        &self,
        out: &Tensor<CpuRuntime>,
        planes: YuvPlanes<'_, CpuRuntime>,
        spec: &YuvSpec,
        layout: ConvLayout,
    ) -> Result<()> {
        let op = "yuv_to_rgb_into";
        let frame = plan_decode_into(out, &planes, spec, layout, color_float_ok, op)?;
        yuv_to_rgb_run(self, &frame, out, layout)
    }

    fn rgb_to_yuv_into(
        &self,
        rgb: &Tensor<CpuRuntime>,
        planes: YuvPlanes<'_, CpuRuntime>,
        spec: &YuvSpec,
        layout: ConvLayout,
    ) -> Result<()> {
        let op = "rgb_to_yuv_into";
        let frame = plan_encode(rgb, &planes, spec, layout, color_float_ok, op)?;
        rgb_to_yuv_run(self, &frame, rgb, layout)
    }

    fn rgb24_to_float(
        &self,
        rgb24: &Tensor<CpuRuntime>,
        out_dtype: DType,
        layout: ConvLayout,
    ) -> Result<Tensor<CpuRuntime>> {
        let [h, w] = plan_unpack(rgb24, out_dtype, color_float_ok, "rgb24_to_float")?;
        let out = Tensor::<CpuRuntime>::empty(&rgb_shape(layout, h, w), out_dtype, &self.device)?;
        rgb24_to_float_run(self, rgb24, &out, layout, [h, w])?;
        Ok(out)
    }

    fn rgb24_to_float_into(
        &self,
        out: &Tensor<CpuRuntime>,
        rgb24: &Tensor<CpuRuntime>,
        layout: ConvLayout,
    ) -> Result<()> {
        let op = "rgb24_to_float_into";
        let size = plan_unpack_into(out, rgb24, layout, color_float_ok, op)?;
        rgb24_to_float_run(self, rgb24, out, layout, size)
    }

    fn float_to_rgb24(
        &self,
        rgb: &Tensor<CpuRuntime>,
        layout: ConvLayout,
    ) -> Result<Tensor<CpuRuntime>> {
        let [h, w] = plan_pack(rgb, layout, color_float_ok, "float_to_rgb24")?;
        let out = Tensor::<CpuRuntime>::empty(&[h, w, 3], DType::U8, &self.device)?;
        float_to_rgb24_run(self, rgb, &out, layout, [h, w])?;
        Ok(out)
    }

    fn float_to_rgb24_into(
        &self,
        out: &Tensor<CpuRuntime>,
        rgb: &Tensor<CpuRuntime>,
        layout: ConvLayout,
    ) -> Result<()> {
        let op = "float_to_rgb24_into";
        let size = plan_pack_into(out, rgb, layout, color_float_ok, op)?;
        float_to_rgb24_run(self, rgb, out, layout, size)
    }
}
