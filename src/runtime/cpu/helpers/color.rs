//! CPU dtype dispatch of the `ColorOps` kernels.
//!
//! Every view passes to its kernel with its strides, so no operand is copied
//! to contiguous first.

use super::super::kernels::{
    PlaneSample, Rgb24Geometry, float_to_rgb24_kernel, rgb_to_yuv_kernel, rgb24_to_float_kernel,
    yuv_to_rgb_kernel,
};
use super::super::{CpuClient, CpuRuntime};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::ops::ConvLayout;
use crate::ops::common::{YuvFrame, rgb_strides, rgb24_strides};
use crate::tensor::Tensor;

/// Float dtypes the CPU colour kernels read and write.
pub fn color_float_ok(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64)
        || (cfg!(feature = "f16") && matches!(dtype, DType::F16 | DType::BF16))
}

/// Run `$body` with `$t` bound to the element type of float dtype `$dtype`.
macro_rules! with_float {
    ($dtype:expr, $op:expr, $t:ident => $body:expr) => {
        match $dtype {
            DType::F32 => {
                type $t = f32;
                $body
            }
            DType::F64 => {
                type $t = f64;
                $body
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                type $t = half::f16;
                $body
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                type $t = half::bf16;
                $body
            }
            dtype => return Err(Error::UnsupportedDType { dtype, op: $op }),
        }
    };
}
pub(crate) use with_float;

/// # Safety
///
/// As [`yuv_to_rgb_kernel`], with `P` the frame's sample type.
unsafe fn decode_samples<P: PlaneSample, O: Element>(
    client: &CpuClient,
    frame: &YuvFrame,
    out: &Tensor<CpuRuntime>,
    layout: ConvLayout,
) {
    let strides = rgb_strides(out.strides(), layout);
    // SAFETY: forwarded from this function's contract.
    unsafe { yuv_to_rgb_kernel::<P, O>(client, frame, out.ptr() as *mut O, strides) }
}

/// Convert `frame` into the float RGB view `out`, which passed the checks.
pub fn yuv_to_rgb_run(
    client: &CpuClient,
    frame: &YuvFrame,
    out: &Tensor<CpuRuntime>,
    layout: ConvLayout,
) -> Result<()> {
    with_float!(out.dtype(), "yuv_to_rgb", T => {
        // SAFETY: `plan_yuv_frame` built `frame` from the plane views, and
        // `out` has the frame's RGB shape and passed the repeat and overlap
        // checks.
        unsafe {
            if frame.wide {
                decode_samples::<u16, T>(client, frame, out, layout);
            } else {
                decode_samples::<u8, T>(client, frame, out, layout);
            }
        }
    });
    Ok(())
}

/// Convert the float RGB view `rgb` into the planes of `frame`, which passed
/// the checks.
pub fn rgb_to_yuv_run(
    client: &CpuClient,
    frame: &YuvFrame,
    rgb: &Tensor<CpuRuntime>,
    layout: ConvLayout,
) -> Result<()> {
    let strides = rgb_strides(rgb.strides(), layout);
    with_float!(rgb.dtype(), "rgb_to_yuv", T => {
        let src = rgb.ptr() as *const T;
        // SAFETY: `plan_yuv_frame` built `frame` from the plane views, `rgb`
        // has the frame's RGB shape, and the planes passed the repeat and
        // overlap checks.
        unsafe {
            if frame.wide {
                rgb_to_yuv_kernel::<T, u16>(client, frame, src, strides);
            } else {
                rgb_to_yuv_kernel::<T, u8>(client, frame, src, strides);
            }
        }
    });
    Ok(())
}

/// Unpack the U8 `[H, W, 3]` view `rgb24` into the float RGB view `out`.
pub fn rgb24_to_float_run(
    client: &CpuClient,
    rgb24: &Tensor<CpuRuntime>,
    out: &Tensor<CpuRuntime>,
    layout: ConvLayout,
    size: [usize; 2],
) -> Result<()> {
    let g = Rgb24Geometry {
        size,
        src_strides: rgb24_strides(rgb24.strides()),
        dst_strides: rgb_strides(out.strides(), layout),
    };
    with_float!(out.dtype(), "rgb24_to_float", T => {
        // SAFETY: both views have the checked shapes, and `out` passed the
        // repeat and overlap checks.
        unsafe {
            rgb24_to_float_kernel::<T>(client, rgb24.ptr() as *const u8, out.ptr() as *mut T, g)
        }
    });
    Ok(())
}

/// Pack the float RGB view `rgb` into the U8 `[H, W, 3]` view `out`.
pub fn float_to_rgb24_run(
    client: &CpuClient,
    rgb: &Tensor<CpuRuntime>,
    out: &Tensor<CpuRuntime>,
    layout: ConvLayout,
    size: [usize; 2],
) -> Result<()> {
    let g = Rgb24Geometry {
        size,
        src_strides: rgb_strides(rgb.strides(), layout),
        dst_strides: rgb24_strides(out.strides()),
    };
    with_float!(rgb.dtype(), "float_to_rgb24", T => {
        // SAFETY: both views have the checked shapes, and `out` passed the
        // repeat and overlap checks.
        unsafe {
            float_to_rgb24_kernel::<T>(client, rgb.ptr() as *const T, out.ptr() as *mut u8, g)
        }
    });
    Ok(())
}
