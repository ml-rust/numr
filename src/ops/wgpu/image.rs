//! WebGPU implementation of pixel shuffle, pixel unshuffle and nearest upsampling.

use crate::error::Result;
use crate::ops::common::{plan_pixel_shuffle, plan_pixel_unshuffle, plan_upsample_nearest2d};
use crate::ops::{ConvLayout, ImageOps, NearestSize};
use crate::runtime::wgpu::ops::native::{native_image, native_image_into};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// ImageOps implementation for WebGPU runtime. F32, I32 and U32.
impl ImageOps<WgpuRuntime> for WgpuClient {
    fn pixel_shuffle(
        &self,
        input: &Tensor<WgpuRuntime>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_image(self, &plan_pixel_shuffle(input, factor, layout)?, input)
    }

    fn pixel_shuffle_into(
        &self,
        out: &Tensor<WgpuRuntime>,
        input: &Tensor<WgpuRuntime>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<()> {
        native_image_into(
            self,
            &plan_pixel_shuffle(input, factor, layout)?,
            out,
            input,
        )
    }

    fn pixel_unshuffle(
        &self,
        input: &Tensor<WgpuRuntime>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_image(self, &plan_pixel_unshuffle(input, factor, layout)?, input)
    }

    fn pixel_unshuffle_into(
        &self,
        out: &Tensor<WgpuRuntime>,
        input: &Tensor<WgpuRuntime>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<()> {
        native_image_into(
            self,
            &plan_pixel_unshuffle(input, factor, layout)?,
            out,
            input,
        )
    }

    fn upsample_nearest2d(
        &self,
        input: &Tensor<WgpuRuntime>,
        size: NearestSize,
        layout: ConvLayout,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_image(self, &plan_upsample_nearest2d(input, size, layout)?, input)
    }

    fn upsample_nearest2d_into(
        &self,
        out: &Tensor<WgpuRuntime>,
        input: &Tensor<WgpuRuntime>,
        size: NearestSize,
        layout: ConvLayout,
    ) -> Result<()> {
        native_image_into(
            self,
            &plan_upsample_nearest2d(input, size, layout)?,
            out,
            input,
        )
    }
}
