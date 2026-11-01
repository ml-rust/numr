//! Launchers for the COO sort kernels in `sparse_coo.cu`.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::CudaContext;
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;
use crate::runtime::cuda::kernels::loader::{
    BLOCK_SIZE, LaunchConfig, get_kernel_function, get_or_load_module, kernel_names, launch_config,
};

fn linear_config(n: usize) -> LaunchConfig {
    let grid_size = (n as u32).div_ceil(BLOCK_SIZE);
    launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0)
}

fn launch_failed(kernel: &str, n: usize, e: impl std::fmt::Debug) -> Error {
    Error::Internal(format!(
        "CUDA {kernel} launch over {n} entries failed: {e:?}"
    ))
}

/// `keys[i] = primary[i] * extent + secondary[i]` over `nnz` I64 entries.
///
/// # Safety
///
/// `primary`, `secondary` and `keys_out` must be live device buffers of
/// `nnz` I64 elements on `context`.
pub unsafe fn launch_coo_compute_keys(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    primary: u64,
    secondary: u64,
    keys_out: u64,
    extent: i64,
    nnz: usize,
) -> Result<()> {
    let kernel = "coo_compute_keys_i64";
    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel)?;
    let nnz_u32 = nnz as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&primary);
    builder.arg(&secondary);
    builder.arg(&keys_out);
    builder.arg(&extent);
    builder.arg(&nnz_u32);
    unsafe { builder.launch(linear_config(nnz)) }.map_err(|e| launch_failed(kernel, nnz, e))?;
    Ok(())
}

/// Stable sort of `n` (I64 key, I32 id) pairs by key, in place. Equal keys
/// keep their input order.
///
/// # Safety
///
/// `keys` must hold `n` I64 elements and `indices` `n` I32 elements, both
/// live device buffers on `context`.
pub unsafe fn launch_thrust_sort_pairs_i64_i32(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    keys: u64,
    indices: u64,
    n: u32,
) -> Result<()> {
    let kernel = "thrust_sort_pairs_i64_i32_kernel";
    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel)?;
    // One thread: Thrust runs the sort from that thread.
    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let mut builder = stream.launch_builder(&function);
    builder.arg(&keys);
    builder.arg(&indices);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| launch_failed(kernel, n as usize, e))?;
    Ok(())
}

/// `indices = [0, 1, ..., n - 1]` as I32.
///
/// # Safety
///
/// `indices` must be a live device buffer of `n` I32 elements on `context`.
pub unsafe fn launch_coo_init_indices(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    indices: u64,
    n: usize,
) -> Result<()> {
    let kernel = "coo_init_indices_i32";
    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel)?;
    let n_u32 = n as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&indices);
    builder.arg(&n_u32);
    unsafe { builder.launch(linear_config(n)) }.map_err(|e| launch_failed(kernel, n, e))?;
    Ok(())
}

/// `values_out[i] = values_in[indices[i]]` for `n` elements of `dtype`.
///
/// The kernel moves raw words, so any dtype of 1, 2, 4 or 8 bytes works.
/// Wider dtypes return `UnsupportedDType`.
///
/// # Safety
///
/// `values_in` must be a live device buffer of `dtype` holding every index in
/// `indices`. `indices` must hold `n` I32 elements and `values_out` `n`
/// elements of `dtype`.
pub unsafe fn launch_coo_gather(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    values_in: u64,
    indices: u64,
    values_out: u64,
    n: usize,
) -> Result<()> {
    let kernel = match dtype.size_in_bytes() {
        1 => "coo_gather_w1",
        2 => "coo_gather_w2",
        4 => "coo_gather_w4",
        8 => "coo_gather_w8",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "COO gather (CUDA moves elements of 1, 2, 4 or 8 bytes)",
            });
        }
    };
    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_COO_MODULE)?;
    let function = get_kernel_function(&module, kernel)?;
    let n_u32 = n as u32;
    let mut builder = stream.launch_builder(&function);
    builder.arg(&values_in);
    builder.arg(&indices);
    builder.arg(&values_out);
    builder.arg(&n_u32);
    unsafe { builder.launch(linear_config(n)) }.map_err(|e| launch_failed(kernel, n, e))?;
    Ok(())
}
