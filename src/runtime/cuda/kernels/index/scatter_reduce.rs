//! Scatter-with-reduction kernel launcher.
//!
//! [`launch_scatter_reduce`] groups the source elements by destination with a
//! stable radix sort (see `scatter_reduce_sort.rs`), then reduces each
//! destination's contributions in increasing source position. No step
//! accumulates through a float atomic, so the result is bit-identical run to
//! run and every reduction, `mean` included, is one pipeline for every dtype.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module,
    kernel_names::SCATTER_REDUCE_MODULE, launch_config,
};
use super::dtype_gate::index_dtype_suffix;
use super::scatter_reduce_sort::sort_scatter_keys;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::{CudaDevice, GuardedStream};

/// Scatter reduce operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterReduceOpCuda {
    /// Sum reduction: accumulate values by addition.
    Sum,
    /// Max reduction: keep the maximum value.
    Max,
    /// Min reduction: keep the minimum value.
    Min,
    /// Product reduction: accumulate values by multiplication.
    Prod,
    /// Mean reduction: sum, then divide by the number of contributions.
    Mean,
}

impl ScatterReduceOpCuda {
    /// The kernel-name fragment for this operation.
    fn name(self) -> &'static str {
        match self {
            ScatterReduceOpCuda::Sum => "sum",
            ScatterReduceOpCuda::Max => "max",
            ScatterReduceOpCuda::Min => "min",
            ScatterReduceOpCuda::Prod => "prod",
            ScatterReduceOpCuda::Mean => "mean",
        }
    }
}

/// Launch scatter_reduce.
///
/// `dst_ptr` must already hold the initial value: a copy of the destination
/// when `include_self` is set, otherwise the reduction's identity. Source
/// element `e` lands at its own coordinates with the coordinate on `dim`
/// replaced by `indices[e]`. An index outside `[0, dst_shape[dim])` is
/// skipped. `src_shape` may be smaller than `dst_shape` on any axis but `dim`.
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] for F16, BF16, FP8, Bool and complex
/// dtypes, which have no reduce kernel, and the limits of the key sort.
///
/// # Safety
///
/// `src_ptr` and `indices_ptr` must point to `src_shape.iter().product()`
/// valid elements of `dtype` and i64, and `dst_ptr` to
/// `dst_shape.iter().product()` elements of `dtype`, all in device memory on
/// `stream`'s context.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_scatter_reduce(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    device: &CudaDevice,
    dtype: DType,
    src_ptr: u64,
    indices_ptr: u64,
    dst_ptr: u64,
    src_shape: &[usize],
    dst_shape: &[usize],
    dim: usize,
    op: ScatterReduceOpCuda,
    include_self: bool,
) -> Result<()> {
    if !(matches!(dtype, DType::F32 | DType::F64) || dtype.is_int()) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "scatter_reduce",
        });
    }
    let suffix = index_dtype_suffix(dtype, "scatter_reduce")?;
    let n: usize = src_shape.iter().product();
    let dst_numel: usize = dst_shape.iter().product();
    if n == 0 || dst_numel == 0 {
        return Ok(());
    }

    let sorted = unsafe {
        sort_scatter_keys(
            context,
            stream,
            device_index,
            device,
            indices_ptr,
            src_shape,
            dst_shape,
            dim,
        )?
    };

    let func_name = format!("scatter_reduce_{}_{}", op.name(), suffix);
    unsafe {
        let module = get_or_load_module(context, device_index, SCATTER_REDUCE_MODULE)?;
        let func = get_kernel_function(&module, &func_name)?;
        let cfg = launch_config(elementwise_launch_config(dst_numel)?, (BLOCK_SIZE, 1, 1), 0);

        // sort_scatter_keys has checked both counts against u32::MAX.
        let n_u32 = n as u32;
        let dst_numel_u32 = dst_numel as u32;
        let include_self_u32 = u32::from(include_self);
        let (keys_ptr, vals_ptr) = (sorted.keys.ptr(), sorted.vals.ptr());

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&keys_ptr);
        builder.arg(&vals_ptr);
        builder.arg(&dst_ptr);
        builder.arg(&n_u32);
        builder.arg(&dst_numel_u32);
        builder.arg(&include_self_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA {func_name} kernel launch failed: {e:?}"))
        })?;
    }
    Ok(())
}
