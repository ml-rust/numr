//! Checked CUDA `index_put` kernel dispatch.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, launch_config,
};
use super::dtype_gate::index_dtype_suffix;
use super::gather::INDEX_MODULE;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;

/// Put source values at indices along a dimension.
///
/// # Safety
///
/// - Pointers address valid memory on the supplied stream's device.
/// - `indices_ptr` contains `index_len` contiguous I64 elements.
/// - `src_ptr` contains `outer_size * index_len * inner_size` contiguous elements.
/// - `output_ptr` contains an independent, initialized input copy.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_index_put(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    indices_ptr: u64,
    src_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
    index_len: usize,
) -> Result<()> {
    let suffix = index_dtype_suffix(dtype, "index_put")?;
    let outer_u32 = checked_scalar("outer_size", outer_size)?;
    let dim_u32 = checked_scalar("dim_size", dim_size)?;
    let inner_u32 = checked_scalar("inner_size", inner_size)?;
    let index_len_u32 = checked_scalar("index_len", index_len)?;
    let total = checked_count(outer_size, index_len, inner_size)?;
    checked_count(outer_size, dim_size, inner_size)?;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = format!("index_put_{suffix}");
        let func = get_kernel_function(&module, &func_name)?;
        let grid = elementwise_launch_config(total)?;
        let cfg = launch_config(grid, (BLOCK_SIZE, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&indices_ptr);
        builder.arg(&src_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer_u32);
        builder.arg(&dim_u32);
        builder.arg(&inner_u32);
        builder.arg(&index_len_u32);
        builder.launch(cfg).map_err(|error| {
            Error::Internal(format!("CUDA index_put kernel launch failed: {error:?}"))
        })?;
    }
    Ok(())
}

fn checked_scalar(name: &str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        limitation(format!(
            "{name} {value} exceeds u32::MAX. Reduce the tensor size."
        ))
    })
}

fn checked_count(outer: usize, extent: usize, inner: usize) -> Result<usize> {
    let count = outer
        .checked_mul(extent)
        .and_then(|value| value.checked_mul(inner))
        .ok_or_else(|| {
            limitation("Index dimensions overflow usize. Reduce the tensor size.".into())
        })?;
    checked_scalar("count", count)?;
    Ok(count)
}

fn limitation(reason: String) -> Error {
    Error::BackendLimitation {
        backend: "cuda",
        operation: "index_put",
        reason,
    }
}
