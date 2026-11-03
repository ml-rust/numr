//! Launcher for the `ImageOps` gather kernels in `image.cu`.
//!
//! One thread per output element in a grid-stride loop with 64-bit indices.
//! The element size picks the kernel, so every dtype runs. Strides and extents
//! travel as kernel arguments, so a launch allocates nothing and records into
//! a CUDA graph.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, MAX_GRID_DIM_X, get_kernel_function, get_or_load_module, launch_config,
};
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;

/// Module name of `image.cu`.
pub const IMAGE_MODULE: &str = "image";

/// Which gather kernel a launch runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageKernel {
    /// `pixel_shuffle_b*`.
    PixelShuffle,
    /// `pixel_unshuffle_b*`.
    PixelUnshuffle,
    /// `upsample_nearest2d_b*`.
    UpsampleNearest2d,
}

impl ImageKernel {
    fn base_name(self) -> &'static str {
        match self {
            Self::PixelShuffle => "pixel_shuffle",
            Self::PixelUnshuffle => "pixel_unshuffle",
            Self::UpsampleNearest2d => "upsample_nearest2d",
        }
    }
}

/// Geometry and op parameters of one gather launch.
#[derive(Debug, Clone, Copy)]
pub struct ImageLaunch {
    /// Kernel to run.
    pub kernel: ImageKernel,
    /// Element size in bytes: 1, 2, 4, 8 or 16.
    pub elem_size: usize,
    /// Output extents as `[N, C, H, W]`.
    pub out_dims: [usize; 4],
    /// Source strides in elements, as `[N, C, H, W]`.
    pub src_strides: [isize; 4],
    /// Destination strides in elements, as `[N, C, H, W]`.
    pub dst_strides: [isize; 4],
    /// Decode the output index with C fastest (NHWC order).
    pub channels_last: bool,
    /// Shuffle or unshuffle factor. Unused by the nearest kernel.
    pub factor: usize,
    /// Input `(H, W)`. Used by the nearest kernel only.
    pub in_hw: [usize; 2],
    /// Nearest `(mode, scale)` for H and W. Used by the nearest kernel only.
    pub nearest: [(u32, f32); 2],
}

/// Launch one gather kernel.
///
/// # Errors
///
/// `Error::Internal` for an element size with no kernel or a failed launch.
///
/// # Safety
///
/// - `src_ptr` and `dst_ptr` are device addresses of the first element of
///   each view on the stream's device.
/// - Every address the strides reach lies inside its allocation.
/// - The destination view addresses no element twice and shares no element
///   with the source view.
pub unsafe fn launch_image(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    src_ptr: u64,
    dst_ptr: u64,
    launch: &ImageLaunch,
) -> Result<()> {
    let total: usize = launch.out_dims.iter().product();
    if total == 0 {
        return Ok(());
    }
    if !matches!(launch.elem_size, 1 | 2 | 4 | 8 | 16) {
        return Err(Error::Internal(format!(
            "{}: element size {} has no CUDA gather kernel",
            launch.kernel.base_name(),
            launch.elem_size
        )));
    }

    let module = get_or_load_module(context, device_index, IMAGE_MODULE)?;
    let func_name = format!("{}_b{}", launch.kernel.base_name(), launch.elem_size);
    let func = get_kernel_function(&module, &func_name)?;

    // The grid-stride loop covers any element count, so the grid caps at the
    // largest 1-D extent.
    let blocks = (total as u64).div_ceil(u64::from(BLOCK_SIZE));
    let grid = (blocks.min(u64::from(MAX_GRID_DIM_X)) as u32, 1, 1);
    let cfg = launch_config(grid, (BLOCK_SIZE, 1, 1), 0);

    let total_u64 = total as u64;
    let [_, dim_c, dim_h, dim_w] = launch.out_dims.map(|d| d as u64);
    let ss = launch.src_strides.map(|s| s as i64);
    let ds = launch.dst_strides.map(|s| s as i64);
    let channels_last = u32::from(launch.channels_last);
    let factor = launch.factor as u64;
    let [in_h, in_w] = launch.in_hw.map(|d| d as u64);
    let [(mode_h, scale_h), (mode_w, scale_w)] = launch.nearest;

    // SAFETY: the caller guarantees the addresses and strides. Argument order
    // and types match NUMR_IMAGE_PARAMS in image.cu.
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
        builder.arg(&factor);
        builder.arg(&in_h);
        builder.arg(&in_w);
        builder.arg(&mode_h);
        builder.arg(&mode_w);
        builder.arg(&scale_h);
        builder.arg(&scale_w);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA kernel '{func_name}' launch failed: {e:?}"))
        })?;
    }
    Ok(())
}
