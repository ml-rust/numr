//! Walsh-Hadamard transform CUDA kernel launcher.
//!
//! One block per (row, segment) pair. The segment lives in dynamic shared
//! memory in the accumulator type, so `block_size` is capped by the 48 KB
//! per-block budget every CUDA device guarantees.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, MAX_GRID_DIM_X, get_kernel_function, get_or_load_module, kernel_name, kernel_names,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Dynamic shared memory every CUDA device grants a block without opt-in.
const FWHT_SHARED_MEM_BUDGET: usize = 48 * 1024;

/// Bytes per shared-memory element for `dtype`: the accumulator width, not the
/// storage width. F16 and BF16 accumulate in F32.
fn fwht_acc_bytes(dtype: DType) -> Result<usize> {
    match dtype {
        DType::F32 | DType::F16 | DType::BF16 => Ok(4),
        DType::F64 => Ok(8),
        _ => Err(Error::UnsupportedDType { dtype, op: "fwht" }),
    }
}

/// Largest `block_size` whose segment fits the shared-memory budget for `dtype`.
///
/// 12288 for a 4-byte accumulator, 6144 for an 8-byte one.
pub fn fwht_max_block_size(dtype: DType) -> Result<usize> {
    Ok(FWHT_SHARED_MEM_BUDGET / fwht_acc_bytes(dtype)?)
}

/// Launch the Walsh-Hadamard kernel over every `block_size` segment of every row.
///
/// Grid is `rows * (last_dim / block_size)` blocks of
/// `min(block_size, BLOCK_SIZE)` threads. Each block loads its segment into
/// `block_size * acc_bytes` bytes of dynamic shared memory, multiplies in
/// `signs` and the `1/sqrt(block_size)` scale, runs the butterfly, and stores.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] when `block_size` exceeds
/// [`fwht_max_block_size`] or the grid exceeds `MAX_GRID_DIM_X`.
///
/// # Safety
///
/// - `input_ptr` and `output_ptr` must be valid device memory with at least
///   `rows * last_dim` elements of `dtype`, and must not overlap
/// - `signs_ptr`, when `Some`, must be valid device memory with at least
///   `last_dim` elements of `dtype`
/// - `block_size` must be a nonzero power of two that divides `last_dim`
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of tensors
/// * `input_ptr` - Device pointer to input tensor
/// * `output_ptr` - Device pointer to output tensor
/// * `signs_ptr` - Device pointer to the `last_dim` sign elements, if any
/// * `rows` - Number of independent rows
/// * `last_dim` - Last-dim size (a multiple of `block_size`)
/// * `block_size` - Transform width, a power of two
pub unsafe fn launch_fwht(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    signs_ptr: Option<u64>,
    rows: usize,
    last_dim: usize,
    block_size: usize,
) -> Result<()> {
    let max_block_size = fwht_max_block_size(dtype)?;
    if block_size > max_block_size {
        return Err(Error::InvalidArgument {
            arg: "block_size",
            reason: format!(
                "{block_size} exceeds the CUDA fwht limit of {max_block_size} for {dtype:?} \
                 ({FWHT_SHARED_MEM_BUDGET} bytes of shared memory per block)"
            ),
        });
    }

    let n_segments = last_dim / block_size;
    let grid_size = rows as u64 * n_segments as u64;
    if grid_size > MAX_GRID_DIM_X as u64 {
        return Err(Error::InvalidArgument {
            arg: "rows",
            reason: format!(
                "{rows} rows of {n_segments} segments need a 1-D grid of {grid_size} blocks, \
                 exceeding the CUDA max grid extent of {MAX_GRID_DIM_X}"
            ),
        });
    }

    let module = get_or_load_module(context, device_index, kernel_names::FWHT_MODULE)?;
    let func_name = kernel_name("fwht", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    // `block_size * acc_bytes <= FWHT_SHARED_MEM_BUDGET`, so this fits a u32.
    let shared_mem = (block_size * fwht_acc_bytes(dtype)?) as u32;
    let grid = (grid_size as u32, 1, 1);
    let block = (BLOCK_SIZE.min(block_size as u32), 1, 1);
    let cfg = launch_config(grid, block, shared_mem);

    // A null pointer tells the kernel there are no signs.
    let signs_ptr = signs_ptr.unwrap_or(0);
    let rows_u32 = rows as u32;
    let last_dim_u32 = last_dim as u32;
    let block_size_u32 = block_size as u32;
    let n_segments_u32 = n_segments as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&signs_ptr);
        builder.arg(&rows_u32);
        builder.arg(&last_dim_u32);
        builder.arg(&block_size_u32);
        builder.arg(&n_segments_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA fwht kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_block_size_is_48kb_over_accumulator_width() {
        assert_eq!(fwht_max_block_size(DType::F32).unwrap(), 12288);
        assert_eq!(fwht_max_block_size(DType::F16).unwrap(), 12288);
        assert_eq!(fwht_max_block_size(DType::BF16).unwrap(), 12288);
        assert_eq!(fwht_max_block_size(DType::F64).unwrap(), 6144);
    }

    #[test]
    fn max_block_size_rejects_integer_dtype() {
        assert!(matches!(
            fwht_max_block_size(DType::I32),
            Err(Error::UnsupportedDType { op: "fwht", .. })
        ));
    }
}
