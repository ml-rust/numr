//! CPU implementation of pixel shuffle, pixel unshuffle and nearest upsampling.

use crate::error::Result;
use crate::ops::common::{
    ImagePlan, plan_pixel_shuffle, plan_pixel_unshuffle, plan_upsample_nearest2d,
    validate_image_out,
};
use crate::ops::{ConvLayout, ImageOps, NearestSize};
use crate::runtime::cpu::{CpuClient, CpuRuntime, helpers::image_run};
use crate::tensor::Tensor;

impl CpuClient {
    /// Allocate a contiguous output for `plan` and fill it.
    fn image_alloc(
        &self,
        plan: &ImagePlan,
        input: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        let out = Tensor::<CpuRuntime>::empty(&plan.out_shape, input.dtype(), &self.device)?;
        image_run(self, plan, input, &out)?;
        Ok(out)
    }

    /// Check `out` against `plan` and fill it.
    fn image_into(
        &self,
        plan: &ImagePlan,
        out: &Tensor<CpuRuntime>,
        input: &Tensor<CpuRuntime>,
    ) -> Result<()> {
        validate_image_out(out, plan, input)?;
        image_run(self, plan, input, out)
    }
}

/// ImageOps implementation for CPU runtime. Every dtype.
impl ImageOps<CpuRuntime> for CpuClient {
    fn pixel_shuffle(
        &self,
        input: &Tensor<CpuRuntime>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<Tensor<CpuRuntime>> {
        self.image_alloc(&plan_pixel_shuffle(input, factor, layout)?, input)
    }

    fn pixel_shuffle_into(
        &self,
        out: &Tensor<CpuRuntime>,
        input: &Tensor<CpuRuntime>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<()> {
        self.image_into(&plan_pixel_shuffle(input, factor, layout)?, out, input)
    }

    fn pixel_unshuffle(
        &self,
        input: &Tensor<CpuRuntime>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<Tensor<CpuRuntime>> {
        self.image_alloc(&plan_pixel_unshuffle(input, factor, layout)?, input)
    }

    fn pixel_unshuffle_into(
        &self,
        out: &Tensor<CpuRuntime>,
        input: &Tensor<CpuRuntime>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<()> {
        self.image_into(&plan_pixel_unshuffle(input, factor, layout)?, out, input)
    }

    fn upsample_nearest2d(
        &self,
        input: &Tensor<CpuRuntime>,
        size: NearestSize,
        layout: ConvLayout,
    ) -> Result<Tensor<CpuRuntime>> {
        self.image_alloc(&plan_upsample_nearest2d(input, size, layout)?, input)
    }

    fn upsample_nearest2d_into(
        &self,
        out: &Tensor<CpuRuntime>,
        input: &Tensor<CpuRuntime>,
        size: NearestSize,
        layout: ConvLayout,
    ) -> Result<()> {
        self.image_into(&plan_upsample_nearest2d(input, size, layout)?, out, input)
    }
}
