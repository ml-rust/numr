//! The grouping half of scatter_reduce: every source element's destination
//! key, then a stable LSD radix sort of the keys with the source positions as
//! values. After it, each destination's contributions sit in one run of the
//! sorted keys, in increasing source position.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module,
    kernel_names::SCATTER_REDUCE_MODULE, launch_config,
};
use super::gather::MAX_DIMS;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::common::collapse_scatter_axes;
use crate::runtime::cuda::{CudaDevice, CudaRuntime, GuardedStream};
use crate::tensor::Tensor;

/// Digits per radix pass. Must match SR_RADIX in scatter_reduce.cu.
const RADIX: usize = 256;
/// Bits per radix pass.
const RADIX_BITS: u32 = 8;
/// Threads per block of the radix kernels. Must match SR_THREADS.
const RADIX_THREADS: u32 = 256;
/// Source entries per radix tile. Must match SR_TILE.
const RADIX_TILE: usize = 4096;

/// Keys and source positions, sorted by key. Each run of one key lists its
/// source positions in increasing order.
pub(super) struct SortedKeys {
    pub keys: Tensor<CudaRuntime>,
    pub vals: Tensor<CudaRuntime>,
}

fn launch_error(kernel: &str, e: impl std::fmt::Debug) -> Error {
    Error::Internal(format!("CUDA {kernel} kernel launch failed: {e:?}"))
}

/// Writes each source element's destination key and sorts the keys.
///
/// An index outside `[0, dst_shape[dim])` gets key `dst_numel`, past every
/// real destination.
///
/// # Errors
///
/// Returns [`Error::BackendLimitation`] when the source holds more than
/// `u32::MAX` elements, the destination `u32::MAX` or more, or the shapes keep
/// more than `MAX_DIMS` axes after collapsing.
///
/// # Safety
///
/// `indices_ptr` must point to `src_shape.iter().product()` valid i64 values
/// in device memory on `stream`'s context.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn sort_scatter_keys(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    device: &CudaDevice,
    indices_ptr: u64,
    src_shape: &[usize],
    dst_shape: &[usize],
    dim: usize,
) -> Result<SortedKeys> {
    let n: usize = src_shape.iter().product();
    let dst_numel: usize = dst_shape.iter().product();
    let limitation = |reason: String| Error::BackendLimitation {
        backend: "CUDA",
        operation: "scatter_reduce",
        reason,
    };
    if n > u32::MAX as usize {
        return Err(limitation(format!(
            "source has {n} elements; the kernel addresses at most {}",
            u32::MAX
        )));
    }
    if dst_numel >= u32::MAX as usize {
        return Err(limitation(format!(
            "destination has {dst_numel} elements; the kernel addresses fewer than {}",
            u32::MAX
        )));
    }
    let (axes, dim_pos) = collapse_scatter_axes(src_shape, dst_shape, dim);
    if axes.len() > MAX_DIMS {
        return Err(limitation(format!(
            "source shape {src_shape:?} against destination shape {dst_shape:?} keeps {} axes; \
             the kernel walks at most {MAX_DIMS}",
            axes.len()
        )));
    }

    let mut keys = Tensor::<CudaRuntime>::empty(&[n], DType::U32, device)?;
    let mut vals = Tensor::<CudaRuntime>::empty(&[n], DType::U32, device)?;
    let mut keys_alt = Tensor::<CudaRuntime>::empty(&[n], DType::U32, device)?;
    let mut vals_alt = Tensor::<CudaRuntime>::empty(&[n], DType::U32, device)?;
    let tiles = n.div_ceil(RADIX_TILE);
    let hist = Tensor::<CudaRuntime>::empty(&[RADIX * tiles], DType::U32, device)?;
    let totals = Tensor::<CudaRuntime>::empty(&[RADIX], DType::U32, device)?;

    let n_u32 = n as u32;
    let tiles_u32 = tiles as u32;
    let invalid_key = dst_numel as u32;

    unsafe {
        let module = get_or_load_module(context, device_index, SCATTER_REDUCE_MODULE)?;

        let func = get_kernel_function(&module, "scatter_reduce_keys")?;
        let cfg = launch_config(elementwise_launch_config(n)?, (BLOCK_SIZE, 1, 1), 0);
        let mut src_ext = [1u32; MAX_DIMS];
        let mut dst_strides = [0u32; MAX_DIMS];
        for (i, axis) in axes.iter().enumerate() {
            src_ext[i] = axis.src_extent as u32;
            dst_strides[i] = axis.dst_stride as u32;
        }
        let ndim_u32 = axes.len() as u32;
        let dim_u32 = dim_pos as u32;
        let dim_size_u32 = dst_shape[dim] as u32;
        let (keys_ptr, vals_ptr) = (keys.ptr(), vals.ptr());
        let mut builder = stream.launch_builder(&func);
        builder.arg(&indices_ptr);
        builder.arg(&keys_ptr);
        builder.arg(&vals_ptr);
        builder.arg(&n_u32);
        builder.arg(&ndim_u32);
        builder.arg(&dim_u32);
        builder.arg(&dim_size_u32);
        builder.arg(&invalid_key);
        for v in &src_ext {
            builder.arg(v);
        }
        for v in &dst_strides {
            builder.arg(v);
        }
        builder
            .launch(cfg)
            .map_err(|e| launch_error("scatter_reduce_keys", e))?;

        let hist_fn = get_kernel_function(&module, "scatter_reduce_radix_hist")?;
        let scan_fn = get_kernel_function(&module, "scatter_reduce_radix_scan")?;
        let scatter_fn = get_kernel_function(&module, "scatter_reduce_radix_scatter")?;
        let tile_cfg = launch_config((tiles_u32, 1, 1), (RADIX_THREADS, 1, 1), 0);
        let digit_cfg = launch_config((RADIX as u32, 1, 1), (RADIX_THREADS, 1, 1), 0);
        let (hist_ptr, totals_ptr) = (hist.ptr(), totals.ptr());

        // The largest key is `invalid_key`, so its bit length bounds the passes.
        let key_bits = u32::BITS - invalid_key.leading_zeros();
        let passes = key_bits.div_ceil(RADIX_BITS).max(1);
        for pass in 0..passes {
            let shift = pass * RADIX_BITS;
            let (keys_in, vals_in) = (keys.ptr(), vals.ptr());
            let (keys_out, vals_out) = (keys_alt.ptr(), vals_alt.ptr());

            let mut builder = stream.launch_builder(&hist_fn);
            builder.arg(&keys_in);
            builder.arg(&n_u32);
            builder.arg(&shift);
            builder.arg(&tiles_u32);
            builder.arg(&hist_ptr);
            builder
                .launch(tile_cfg)
                .map_err(|e| launch_error("scatter_reduce_radix_hist", e))?;

            let mut builder = stream.launch_builder(&scan_fn);
            builder.arg(&hist_ptr);
            builder.arg(&tiles_u32);
            builder.arg(&totals_ptr);
            builder
                .launch(digit_cfg)
                .map_err(|e| launch_error("scatter_reduce_radix_scan", e))?;

            let mut builder = stream.launch_builder(&scatter_fn);
            builder.arg(&keys_in);
            builder.arg(&vals_in);
            builder.arg(&n_u32);
            builder.arg(&shift);
            builder.arg(&tiles_u32);
            builder.arg(&hist_ptr);
            builder.arg(&totals_ptr);
            builder.arg(&keys_out);
            builder.arg(&vals_out);
            builder
                .launch(tile_cfg)
                .map_err(|e| launch_error("scatter_reduce_radix_scatter", e))?;

            std::mem::swap(&mut keys, &mut keys_alt);
            std::mem::swap(&mut vals, &mut vals_alt);
        }
    }

    Ok(SortedKeys { keys, vals })
}
