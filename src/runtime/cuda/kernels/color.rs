//! Launchers for the `ColorOps` kernels in `color.cu`.
//!
//! Each kernel takes one `#[repr(C)]` argument struct by value, so a launch
//! uploads nothing and records into a CUDA graph. Every kernel runs a
//! grid-stride loop with 64-bit indices.

use cudarc::driver::safe::CudaContext;
use cudarc::driver::{DeviceRepr, PushKernelArg};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, MAX_GRID_DIM_X, dtype_suffix, get_kernel_function, get_or_load_module,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::common::{ColorCoeffs, PlaneView, YuvFrame};
use crate::runtime::cuda::GuardedStream;

/// Module name of `color.cu`.
pub const COLOR_MODULE: &str = "color";

/// One plane, as `ColorPlane` in `color.cu`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code, reason = "only color.cu reads the fields")]
struct ColorPlaneArg {
    addr: u64,
    row_stride: i64,
    col_stride: i64,
}

impl From<PlaneView> for ColorPlaneArg {
    fn from(p: PlaneView) -> Self {
        Self {
            addr: p.addr,
            row_stride: p.row_stride as i64,
            col_stride: p.col_stride as i64,
        }
    }
}

/// Arguments of a YCbCr kernel, as `YuvArgs` in `color.cu`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code, reason = "only color.cu reads the fields")]
struct YuvKernelArgs {
    y: ColorPlaneArg,
    u: ColorPlaneArg,
    v: ColorPlaneArg,
    rgb: u64,
    rgb_c: i64,
    rgb_h: i64,
    rgb_w: i64,
    height: i64,
    width: i64,
    k: ColorCoeffs,
    shift: u32,
    center: u32,
}

// SAFETY: `#[repr(C)]` plain data whose layout matches `YuvArgs` in color.cu.
unsafe impl DeviceRepr for YuvKernelArgs {}

/// Arguments of an RGB24 kernel, as `Rgb24Args` in `color.cu`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code, reason = "only color.cu reads the fields")]
struct Rgb24KernelArgs {
    src: u64,
    dst: u64,
    src_strides: [i64; 3],
    dst_strides: [i64; 3],
    height: i64,
    width: i64,
}

// SAFETY: `#[repr(C)]` plain data whose layout matches `Rgb24Args` in color.cu.
unsafe impl DeviceRepr for Rgb24KernelArgs {}

/// Direction of a YCbCr launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YuvDirection {
    /// Planes to RGB: `yuv_to_rgb_{p}_{o}`.
    ToRgb,
    /// RGB to planes: `rgb_to_yuv_{i}_{p}`.
    ToYuv,
}

/// Direction of an RGB24 launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rgb24Direction {
    /// RGB24 to float: `rgb24_to_float_{o}`.
    Unpack,
    /// Float to RGB24: `float_to_rgb24_{i}`.
    Pack,
}

/// Kernel suffix of a float dtype, or `None`.
fn float_suffix(dtype: DType) -> Option<&'static str> {
    matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16).then(|| dtype_suffix(dtype))
}

/// Kernel name of a YCbCr launch, or `None` for a dtype with no kernel.
fn yuv_kernel_name(direction: YuvDirection, wide: bool, float: DType) -> Option<String> {
    let f = float_suffix(float)?;
    let p = if wide { "u16" } else { "u8" };
    Some(match direction {
        YuvDirection::ToRgb => format!("yuv_to_rgb_{p}_{f}"),
        YuvDirection::ToYuv => format!("rgb_to_yuv_{f}_{p}"),
    })
}

/// Kernel name of an RGB24 launch, or `None` for a dtype with no kernel.
fn rgb24_kernel_name(direction: Rgb24Direction, float: DType) -> Option<String> {
    let f = float_suffix(float)?;
    Some(match direction {
        Rgb24Direction::Unpack => format!("rgb24_to_float_{f}"),
        Rgb24Direction::Pack => format!("float_to_rgb24_{f}"),
    })
}

/// Load `name`, size a grid-stride grid for `total` threads and launch it.
///
/// # Safety
///
/// `args` matches the parameter struct of kernel `name`, and every address
/// in it is valid for the kernel.
unsafe fn launch<A: DeviceRepr>(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    name: &str,
    total: u64,
    args: &A,
) -> Result<()> {
    if total == 0 {
        return Ok(());
    }
    let module = get_or_load_module(context, device_index, COLOR_MODULE)?;
    let func = get_kernel_function(&module, name)?;
    // The grid-stride loop covers any count, so the grid caps at the largest
    // 1-D extent.
    let blocks = total.div_ceil(u64::from(BLOCK_SIZE));
    let grid = (blocks.min(u64::from(MAX_GRID_DIM_X)) as u32, 1, 1);
    let cfg = launch_config(grid, (BLOCK_SIZE, 1, 1), 0);
    // SAFETY: forwarded from this function's contract.
    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(args);
        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA kernel '{name}' launch failed: {e:?}")))?;
    }
    Ok(())
}

/// Launch a YCbCr kernel between `frame` and the float RGB view at
/// `rgb_ptr` with `[C, H, W]` strides `rgb_strides`.
///
/// # Errors
///
/// `Error::UnsupportedDType` for a float dtype with no kernel, and
/// `Error::Internal` for a failed launch.
///
/// # Safety
///
/// - `frame` came from `plan_yuv_frame` on views of the stream's device.
/// - `rgb_ptr` is the device address of RGB element `[0, 0, 0]`, and every
///   address the strides reach lies inside its allocation.
/// - The written views address no element twice and share no byte with
///   another operand.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_yuv(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    direction: YuvDirection,
    frame: &YuvFrame,
    float: DType,
    rgb_ptr: u64,
    rgb_strides: [isize; 3],
) -> Result<()> {
    let op = match direction {
        YuvDirection::ToRgb => "yuv_to_rgb",
        YuvDirection::ToYuv => "rgb_to_yuv",
    };
    let name = yuv_kernel_name(direction, frame.wide, float)
        .ok_or(Error::UnsupportedDType { dtype: float, op })?;
    let total = match direction {
        YuvDirection::ToRgb => frame.pixels(),
        YuvDirection::ToYuv => frame.pixels() / 4,
    } as u64;
    let [rgb_c, rgb_h, rgb_w] = rgb_strides.map(|s| s as i64);
    let args = YuvKernelArgs {
        y: frame.y.into(),
        u: frame.u.into(),
        v: frame.v.into(),
        rgb: rgb_ptr,
        rgb_c,
        rgb_h,
        rgb_w,
        height: frame.height as i64,
        width: frame.width as i64,
        k: frame.coeffs,
        shift: frame.shift,
        center: u32::from(frame.center),
    };
    // SAFETY: `args` is the `YuvArgs` of every YCbCr kernel, and the caller
    // guarantees the addresses.
    unsafe { launch(context, stream, device_index, &name, total, &args) }
}

/// Geometry of one RGB24 launch.
#[derive(Debug, Clone, Copy)]
pub struct Rgb24Launch {
    /// Pack or unpack.
    pub direction: Rgb24Direction,
    /// Dtype of the float side.
    pub float: DType,
    /// Frame size `[H, W]`.
    pub size: [usize; 2],
    /// Device address of source element `[0, 0, 0]`.
    pub src: u64,
    /// Device address of destination element `[0, 0, 0]`.
    pub dst: u64,
    /// Source strides as `[C, H, W]`.
    pub src_strides: [isize; 3],
    /// Destination strides as `[C, H, W]`.
    pub dst_strides: [isize; 3],
}

/// Launch an RGB24 pack or unpack kernel.
///
/// # Errors
///
/// `Error::UnsupportedDType` for a float dtype with no kernel, and
/// `Error::Internal` for a failed launch.
///
/// # Safety
///
/// Both addresses are device addresses of element `[0, 0, 0]` of views on
/// the stream's device. Every address the strides reach lies inside its
/// allocation. The destination addresses no element twice and shares no
/// byte with the source.
pub unsafe fn launch_rgb24(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    l: &Rgb24Launch,
) -> Result<()> {
    let op = match l.direction {
        Rgb24Direction::Unpack => "rgb24_to_float",
        Rgb24Direction::Pack => "float_to_rgb24",
    };
    let name = rgb24_kernel_name(l.direction, l.float)
        .ok_or(Error::UnsupportedDType { dtype: l.float, op })?;
    let [h, w] = l.size;
    let args = Rgb24KernelArgs {
        src: l.src,
        dst: l.dst,
        src_strides: l.src_strides.map(|s| s as i64),
        dst_strides: l.dst_strides.map(|s| s as i64),
        height: h as i64,
        width: w as i64,
    };
    let total = (h * w * 3) as u64;
    // SAFETY: `args` is the `Rgb24Args` of every RGB24 kernel, and the
    // caller guarantees the addresses.
    unsafe { launch(context, stream, device_index, &name, total, &args) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_names_cover_every_dtype_pair() {
        assert_eq!(
            yuv_kernel_name(YuvDirection::ToRgb, true, DType::BF16).as_deref(),
            Some("yuv_to_rgb_u16_bf16")
        );
        assert_eq!(
            yuv_kernel_name(YuvDirection::ToYuv, false, DType::F64).as_deref(),
            Some("rgb_to_yuv_f64_u8")
        );
        assert_eq!(
            rgb24_kernel_name(Rgb24Direction::Pack, DType::F16).as_deref(),
            Some("float_to_rgb24_f16")
        );
        assert_eq!(
            yuv_kernel_name(YuvDirection::ToRgb, false, DType::I32),
            None
        );
    }

    #[test]
    fn argument_structs_match_the_cuda_layout() {
        assert_eq!(std::mem::size_of::<ColorCoeffs>(), 64);
        assert_eq!(std::mem::size_of::<YuvKernelArgs>(), 192);
        assert_eq!(std::mem::size_of::<Rgb24KernelArgs>(), 80);
    }
}
