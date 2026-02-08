//! CUDA kernel launchers for quasi-random sequence generation

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_names,
    launch_config,
};
use crate::error::{Error, Result};
use crate::ops::common::quasirandom::compute_all_direction_vectors;
use crate::runtime::Runtime;
use crate::runtime::cuda::{CudaDevice, CudaRuntime};
use cudarc::driver::{CudaContext, CudaStream, PushKernelArg};
use std::sync::Arc;

/// Launch Sobol sequence generation kernel (F32).
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `n_points * dimension` elements
pub unsafe fn launch_sobol_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &CudaDevice,
    out_ptr: u64,
    n_points: usize,
    dimension: usize,
    skip: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::QUASIRANDOM_MODULE)?;
    let func_name = "sobol_f32";
    let func = get_kernel_function(&module, func_name)?;

    // Compute direction vectors on host and upload to GPU
    let direction_vectors = compute_all_direction_vectors(dimension);
    let dv_bytes = bytemuck::cast_slice::<u32, u8>(&direction_vectors);
    let dv_ptr = CudaRuntime::allocate(dv_bytes.len(), device)?;
    CudaRuntime::copy_to_device(dv_bytes, dv_ptr, device)?;

    let grid = elementwise_launch_config(n_points);
    let block = (BLOCK_SIZE, 1, 1);
    let n = n_points as u32;
    let dim = dimension as u32;
    let skip_u32 = skip as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&dv_ptr);
        builder.arg(&n);
        builder.arg(&dim);
        builder.arg(&skip_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA sobol_f32 kernel launch failed: {:?}", e))
        })?;
    }

    // Synchronize before deallocating direction vectors
    stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("CUDA stream sync failed: {:?}", e)))?;

    CudaRuntime::deallocate(dv_ptr, dv_bytes.len(), device);

    Ok(())
}

/// Launch Sobol sequence generation kernel (F64).
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `n_points * dimension` elements
pub unsafe fn launch_sobol_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &CudaDevice,
    out_ptr: u64,
    n_points: usize,
    dimension: usize,
    skip: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::QUASIRANDOM_MODULE)?;
    let func_name = "sobol_f64";
    let func = get_kernel_function(&module, func_name)?;

    // Compute direction vectors on host and upload to GPU
    let direction_vectors = compute_all_direction_vectors(dimension);
    let dv_bytes = bytemuck::cast_slice::<u32, u8>(&direction_vectors);
    let dv_ptr = CudaRuntime::allocate(dv_bytes.len(), device)?;
    CudaRuntime::copy_to_device(dv_bytes, dv_ptr, device)?;

    let grid = elementwise_launch_config(n_points);
    let block = (BLOCK_SIZE, 1, 1);
    let n = n_points as u32;
    let dim = dimension as u32;
    let skip_u32 = skip as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&dv_ptr);
        builder.arg(&n);
        builder.arg(&dim);
        builder.arg(&skip_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA sobol_f64 kernel launch failed: {:?}", e))
        })?;
    }

    // Synchronize before deallocating direction vectors
    stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("CUDA stream sync failed: {:?}", e)))?;

    CudaRuntime::deallocate(dv_ptr, dv_bytes.len(), device);

    Ok(())
}

/// Launch Halton sequence generation kernel (F32).
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `n_points * dimension` elements
pub unsafe fn launch_halton_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    out_ptr: u64,
    n_points: usize,
    dimension: usize,
    skip: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::QUASIRANDOM_MODULE)?;
    let func_name = "halton_f32";
    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(n_points);
    let block = (BLOCK_SIZE, 1, 1);
    let n = n_points as u32;
    let dim = dimension as u32;
    let skip_u32 = skip as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&dim);
        builder.arg(&skip_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA halton_f32 kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch Halton sequence generation kernel (F64).
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `n_points * dimension` elements
pub unsafe fn launch_halton_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    out_ptr: u64,
    n_points: usize,
    dimension: usize,
    skip: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::QUASIRANDOM_MODULE)?;
    let func_name = "halton_f64";
    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(n_points);
    let block = (BLOCK_SIZE, 1, 1);
    let n = n_points as u32;
    let dim = dimension as u32;
    let skip_u32 = skip as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&dim);
        builder.arg(&skip_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA halton_f64 kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch Latin Hypercube Sampling kernel (F32).
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `n_samples * dimension` elements
pub unsafe fn launch_latin_hypercube_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    out_ptr: u64,
    n_samples: usize,
    dimension: usize,
    seed: u64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::QUASIRANDOM_MODULE)?;
    let func_name = "latin_hypercube_f32";
    let func = get_kernel_function(&module, func_name)?;

    // Each block handles one dimension
    let grid = (dimension as u32, 1, 1);
    let block = (BLOCK_SIZE.min(256), 1, 1); // Use smaller block for shared memory

    // Shared memory for interval shuffling: n_samples * sizeof(u32)
    let shared_mem_bytes = (n_samples * std::mem::size_of::<u32>()) as u32;
    let n = n_samples as u32;
    let dim = dimension as u32;
    let cfg = launch_config(grid, block, shared_mem_bytes);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&dim);
        builder.arg(&seed);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA latin_hypercube_f32 kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch Latin Hypercube Sampling kernel (F64).
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `n_samples * dimension` elements
pub unsafe fn launch_latin_hypercube_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    out_ptr: u64,
    n_samples: usize,
    dimension: usize,
    seed: u64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::QUASIRANDOM_MODULE)?;
    let func_name = "latin_hypercube_f64";
    let func = get_kernel_function(&module, func_name)?;

    // Each block handles one dimension
    let grid = (dimension as u32, 1, 1);
    let block = (BLOCK_SIZE.min(256), 1, 1); // Use smaller block for shared memory

    // Shared memory for interval shuffling: n_samples * sizeof(u32)
    let shared_mem_bytes = (n_samples * std::mem::size_of::<u32>()) as u32;
    let n = n_samples as u32;
    let dim = dimension as u32;
    let cfg = launch_config(grid, block, shared_mem_bytes);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&dim);
        builder.arg(&seed);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA latin_hypercube_f64 kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}
