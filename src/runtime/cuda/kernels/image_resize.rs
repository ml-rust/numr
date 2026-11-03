//! Launcher for the `resize` pass kernels in `image_resize.cu`.
//!
//! One thread per destination element in a grid-stride loop with 64-bit
//! indices. Each thread computes its window and weights from its own index,
//! so a launch uploads nothing and records into a CUDA graph.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, MAX_GRID_DIM_X, dtype_suffix, get_kernel_function, get_or_load_module,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;

/// Module name of `image_resize.cu`.
pub const IMAGE_RESIZE_MODULE: &str = "image_resize";

/// Geometry and window rule of one pass.
#[derive(Debug, Clone, Copy)]
pub struct ResizeLaunch {
    /// Source element dtype.
    pub src_dtype: DType,
    /// Destination element dtype.
    pub dst_dtype: DType,
    /// Destination extents as `[N, C, H, W]`.
    pub dst_dims: [usize; 4],
    /// Source strides in elements, as `[N, C, H, W]`.
    pub src_strides: [isize; 4],
    /// Destination strides in elements, as `[N, C, H, W]`.
    pub dst_strides: [isize; 4],
    /// Decode the destination index with C fastest (NHWC order).
    pub channels_last: bool,
    /// Interpolated canonical axis: 2 for H, 3 for W.
    pub axis: usize,
    /// `ResizeMode::kernel_mode` of the pass.
    pub mode: u32,
    /// Source extent along `axis`.
    pub in_size: usize,
    /// Destination extent along `axis`.
    pub out_size: usize,
}

/// Kernel-name suffix of a float dtype, or `None` for a dtype with no kernel.
fn float_suffix(dtype: DType) -> Option<&'static str> {
    matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16).then(|| dtype_suffix(dtype))
}

/// Kernel name of a pass from `src` to `dst` elements, or `None` for a pair
/// no schedule uses.
fn kernel_name(src: DType, dst: DType) -> Option<String> {
    let pair_exists = matches!(
        (src, dst),
        (DType::F32, DType::F32)
            | (DType::F64, DType::F64)
            | (DType::F16 | DType::BF16, DType::F32)
            | (DType::F32, DType::F16 | DType::BF16)
            | (DType::F16, DType::F16)
            | (DType::BF16, DType::BF16)
    );
    if !pair_exists {
        return None;
    }
    Some(format!(
        "resize_pass_{}_{}",
        float_suffix(src)?,
        float_suffix(dst)?
    ))
}

/// Launch one pass kernel.
///
/// # Errors
///
/// `Error::UnsupportedDType` for an element pair with no kernel, and
/// `Error::Internal` for a failed launch.
///
/// # Safety
///
/// - `src_ptr` and `dst_ptr` are device addresses of the first element of
///   each view on the stream's device.
/// - Every address the strides reach lies inside its allocation.
/// - The destination view addresses no element twice and shares no element
///   with the source view.
pub unsafe fn launch_resize_pass(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    src_ptr: u64,
    dst_ptr: u64,
    launch: &ResizeLaunch,
) -> Result<()> {
    let total: usize = launch.dst_dims.iter().product();
    if total == 0 {
        return Ok(());
    }
    let func_name =
        kernel_name(launch.src_dtype, launch.dst_dtype).ok_or(Error::UnsupportedDType {
            dtype: launch.src_dtype,
            op: "resize",
        })?;
    let module = get_or_load_module(context, device_index, IMAGE_RESIZE_MODULE)?;
    let func = get_kernel_function(&module, &func_name)?;

    // The grid-stride loop covers any element count, so the grid caps at the
    // largest 1-D extent.
    let blocks = (total as u64).div_ceil(u64::from(BLOCK_SIZE));
    let grid = (blocks.min(u64::from(MAX_GRID_DIM_X)) as u32, 1, 1);
    let cfg = launch_config(grid, (BLOCK_SIZE, 1, 1), 0);

    let total_u64 = total as u64;
    let [_, dim_c, dim_h, dim_w] = launch.dst_dims.map(|d| d as u64);
    let ss = launch.src_strides.map(|s| s as i64);
    let ds = launch.dst_strides.map(|s| s as i64);
    let channels_last = u32::from(launch.channels_last);
    let axis = launch.axis as u32;
    let mode = launch.mode;
    let in_size = launch.in_size as i64;
    let out_size = launch.out_size as i64;

    // SAFETY: the caller guarantees the addresses and strides. Argument order
    // and types match NUMR_RESIZE_PARAMS in image_resize.cu.
    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&dst_ptr);
        builder.arg(&total_u64);
        builder.arg(&dim_c);
        builder.arg(&dim_h);
        builder.arg(&dim_w);
        for s in &ss {
            builder.arg(s);
        }
        for s in &ds {
            builder.arg(s);
        }
        builder.arg(&channels_last);
        builder.arg(&axis);
        builder.arg(&mode);
        builder.arg(&in_size);
        builder.arg(&out_size);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA kernel '{func_name}' launch failed: {e:?}"))
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_names_cover_every_schedule_pair() {
        assert_eq!(
            kernel_name(DType::F16, DType::F32).as_deref(),
            Some("resize_pass_f16_f32")
        );
        assert_eq!(
            kernel_name(DType::F32, DType::BF16).as_deref(),
            Some("resize_pass_f32_bf16")
        );
        assert_eq!(
            kernel_name(DType::F64, DType::F64).as_deref(),
            Some("resize_pass_f64_f64")
        );
        assert_eq!(kernel_name(DType::F64, DType::F32), None);
        assert_eq!(kernel_name(DType::I32, DType::I32), None);
    }
}
