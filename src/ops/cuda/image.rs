//! CUDA implementation of pixel shuffle, pixel unshuffle and nearest upsampling.
//!
//! Strided input and output views pass to the kernel with their strides, so
//! no operand is copied to contiguous first and no data leaves the device.

use crate::error::Result;
use crate::ops::common::{
    ImageMap, ImagePlan, canonical_strides, plan_pixel_shuffle, plan_pixel_unshuffle,
    plan_upsample_nearest2d, validate_image_out,
};
use crate::ops::{ConvLayout, ImageOps, NearestSize};
use crate::runtime::cuda::kernels::{ImageKernel, ImageLaunch, launch_image};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl CudaClient {
    /// Run `plan` from `input` into `out`.
    ///
    /// `out` has the planned shape and the input's dtype, addresses each
    /// element once, and shares none with `input`.
    fn image_run(
        &self,
        plan: &ImagePlan,
        input: &Tensor<CudaRuntime>,
        out: &Tensor<CudaRuntime>,
    ) -> Result<()> {
        if plan.numel() == 0 {
            return Ok(());
        }
        let (kernel, factor, nearest) = match plan.map {
            ImageMap::Shuffle { factor } => (ImageKernel::PixelShuffle, factor, [(0, 0.0); 2]),
            ImageMap::Unshuffle { factor } => (ImageKernel::PixelUnshuffle, factor, [(0, 0.0); 2]),
            ImageMap::Nearest { h, w } => (
                ImageKernel::UpsampleNearest2d,
                1,
                [h.kernel_args(), w.kernel_args()],
            ),
        };
        let launch = ImageLaunch {
            kernel,
            elem_size: input.dtype().size_in_bytes(),
            out_dims: plan.out_dims,
            src_strides: canonical_strides(input.strides(), plan.layout),
            dst_strides: canonical_strides(out.strides(), plan.layout),
            channels_last: plan.channels_last(),
            factor,
            in_hw: [plan.in_dims[2], plan.in_dims[3]],
            nearest,
        };
        // SAFETY: `ptr()` folds in each view's offset, the plan matches both
        // shapes, and `out` was checked against repeats and overlap.
        unsafe {
            launch_image(
                &self.context,
                &self.stream,
                self.device.index,
                input.ptr(),
                out.ptr(),
                &launch,
            )
        }
    }

    /// Allocate a contiguous output for `plan` and fill it.
    fn image_alloc(
        &self,
        plan: &ImagePlan,
        input: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let out = Tensor::<CudaRuntime>::empty(&plan.out_shape, input.dtype(), &self.device)?;
        self.image_run(plan, input, &out)?;
        Ok(out)
    }

    /// Check `out` against `plan` and fill it.
    fn image_into(
        &self,
        plan: &ImagePlan,
        out: &Tensor<CudaRuntime>,
        input: &Tensor<CudaRuntime>,
    ) -> Result<()> {
        validate_image_out(out, plan, input)?;
        self.image_run(plan, input, out)
    }
}

/// ImageOps implementation for CUDA runtime. Every dtype.
impl ImageOps<CudaRuntime> for CudaClient {
    fn pixel_shuffle(
        &self,
        input: &Tensor<CudaRuntime>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<Tensor<CudaRuntime>> {
        self.image_alloc(&plan_pixel_shuffle(input, factor, layout)?, input)
    }

    fn pixel_shuffle_into(
        &self,
        out: &Tensor<CudaRuntime>,
        input: &Tensor<CudaRuntime>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<()> {
        self.image_into(&plan_pixel_shuffle(input, factor, layout)?, out, input)
    }

    fn pixel_unshuffle(
        &self,
        input: &Tensor<CudaRuntime>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<Tensor<CudaRuntime>> {
        self.image_alloc(&plan_pixel_unshuffle(input, factor, layout)?, input)
    }

    fn pixel_unshuffle_into(
        &self,
        out: &Tensor<CudaRuntime>,
        input: &Tensor<CudaRuntime>,
        factor: usize,
        layout: ConvLayout,
    ) -> Result<()> {
        self.image_into(&plan_pixel_unshuffle(input, factor, layout)?, out, input)
    }

    fn upsample_nearest2d(
        &self,
        input: &Tensor<CudaRuntime>,
        size: NearestSize,
        layout: ConvLayout,
    ) -> Result<Tensor<CudaRuntime>> {
        self.image_alloc(&plan_upsample_nearest2d(input, size, layout)?, input)
    }

    fn upsample_nearest2d_into(
        &self,
        out: &Tensor<CudaRuntime>,
        input: &Tensor<CudaRuntime>,
        size: NearestSize,
        layout: ConvLayout,
    ) -> Result<()> {
        self.image_into(&plan_upsample_nearest2d(input, size, layout)?, out, input)
    }
}
