//! CUDA kernel launchers for advanced PRNGs

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_names,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::GuardedStream;
use cudarc::driver::{CudaContext, PushKernelArg};
use std::sync::Arc;

/// Get kernel suffix for dtype
#[inline]
fn dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F64 => Ok("f64"),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "advanced_random",
        }),
    }
}

/// Kernel suffix for a Philox dtype: F32, F64, F16 and BF16.
#[inline]
fn philox_dtype_suffix(dtype: DType, op: &'static str) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F64 => Ok("f64"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Element count as the kernel's `unsigned int n`.
#[inline]
fn numel_u32(numel: usize, op: &'static str) -> Result<u32> {
    u32::try_from(numel).map_err(|_| Error::InvalidArgument {
        arg: "shape",
        reason: format!("{op}: {numel} elements exceed the kernel's u32 element count"),
    })
}

// ============================================================================
// Philox Kernels
// ============================================================================

/// Launch Philox uniform kernel
///
/// # Safety
/// - `out_ptr` must be a valid device pointer to `numel` elements
pub unsafe fn launch_philox_uniform(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    key: u64,
    counter: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = format!(
        "philox_uniform_{}",
        philox_dtype_suffix(dtype, "philox_uniform")?
    );
    let n = numel_u32(numel, "philox_uniform")?;
    let module = get_or_load_module(context, device_index, kernel_names::ADVANCED_RANDOM_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    // One thread per 4-element Philox block.
    let grid = elementwise_launch_config(numel.div_ceil(4))?;
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&key);
        builder.arg(&counter);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch Philox randn kernel
///
/// # Safety
/// - `out_ptr` must be a valid device pointer to `numel` elements
pub unsafe fn launch_philox_randn(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    key: u64,
    counter: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = format!(
        "philox_randn_{}",
        philox_dtype_suffix(dtype, "philox_randn")?
    );
    let n = numel_u32(numel, "philox_randn")?;
    let module = get_or_load_module(context, device_index, kernel_names::ADVANCED_RANDOM_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    // One thread per 4-element Philox block.
    let grid = elementwise_launch_config(numel.div_ceil(4))?;
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&key);
        builder.arg(&counter);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// ThreeFry Kernels
// ============================================================================

/// Launch ThreeFry uniform kernel
///
/// # Safety
/// - `out_ptr` must be a valid device pointer to `numel` elements
pub unsafe fn launch_threefry_uniform(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    key: u64,
    counter: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = format!("threefry_uniform_{}", dtype_suffix(dtype)?);
    let n = numel_u32(numel, "threefry_uniform")?;
    let module = get_or_load_module(context, device_index, kernel_names::ADVANCED_RANDOM_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&key);
        builder.arg(&counter);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch ThreeFry randn kernel
///
/// # Safety
/// - `out_ptr` must be a valid device pointer to `numel` elements
pub unsafe fn launch_threefry_randn(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    key: u64,
    counter: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = format!("threefry_randn_{}", dtype_suffix(dtype)?);
    let n = numel_u32(numel, "threefry_randn")?;
    let module = get_or_load_module(context, device_index, kernel_names::ADVANCED_RANDOM_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&key);
        builder.arg(&counter);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// PCG64 Kernels
// ============================================================================

/// Launch PCG64 uniform kernel
///
/// # Safety
/// - `out_ptr` must be a valid device pointer to `numel` elements
pub unsafe fn launch_pcg64_uniform(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    seed: u64,
    stream_id: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = format!("pcg64_uniform_{}", dtype_suffix(dtype)?);
    let n = numel_u32(numel, "pcg64_uniform")?;
    let module = get_or_load_module(context, device_index, kernel_names::ADVANCED_RANDOM_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&seed);
        builder.arg(&stream_id);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch PCG64 randn kernel
///
/// # Safety
/// - `out_ptr` must be a valid device pointer to `numel` elements
pub unsafe fn launch_pcg64_randn(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    seed: u64,
    stream_id: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = format!("pcg64_randn_{}", dtype_suffix(dtype)?);
    let n = numel_u32(numel, "pcg64_randn")?;
    let module = get_or_load_module(context, device_index, kernel_names::ADVANCED_RANDOM_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&seed);
        builder.arg(&stream_id);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// Xoshiro256++ Kernels
// ============================================================================

/// Launch Xoshiro256++ uniform kernel
///
/// # Safety
/// - `out_ptr` must be a valid device pointer to `numel` elements
pub unsafe fn launch_xoshiro256_uniform(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    seed: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = format!("xoshiro256_uniform_{}", dtype_suffix(dtype)?);
    let n = numel_u32(numel, "xoshiro256_uniform")?;
    let module = get_or_load_module(context, device_index, kernel_names::ADVANCED_RANDOM_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&seed);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch Xoshiro256++ randn kernel
///
/// # Safety
/// - `out_ptr` must be a valid device pointer to `numel` elements
pub unsafe fn launch_xoshiro256_randn(
    context: &Arc<CudaContext>,
    stream: &GuardedStream,
    device_index: usize,
    dtype: DType,
    seed: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = format!("xoshiro256_randn_{}", dtype_suffix(dtype)?);
    let n = numel_u32(numel, "xoshiro256_randn")?;
    let module = get_or_load_module(context, device_index, kernel_names::ADVANCED_RANDOM_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&seed);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}
