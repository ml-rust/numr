//! Masked select, masked fill, and broadcast masked operation kernel launchers

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    launch_config,
};
use super::gather::INDEX_MODULE;
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Masked Select
// ============================================================================

/// Launch masked_count kernel to count true elements in mask.
///
/// # Safety
///
/// - mask_ptr must be valid device memory with n u8 elements
/// - count_ptr must be valid device memory with 1 u32 element (initialized to 0)
pub unsafe fn launch_masked_count(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    mask_ptr: u64,
    count_ptr: u64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, "masked_count_kernel")?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&mask_ptr);
        builder.arg(&count_ptr);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA masked_count kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch masked_prefix_sum kernel to compute prefix sum of mask.
///
/// # Safety
///
/// - mask_ptr must be valid device memory with n u8 elements
/// - prefix_sum_ptr must be valid device memory with n u32 elements
pub unsafe fn launch_masked_prefix_sum(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    mask_ptr: u64,
    prefix_sum_ptr: u64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, "masked_prefix_sum_kernel")?;

        let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&mask_ptr);
        builder.arg(&prefix_sum_ptr);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA masked_prefix_sum kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch masked_select kernel.
///
/// Selects elements from input where mask is true, using precomputed prefix sum.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - prefix_sum must be precomputed via launch_masked_prefix_sum
/// - output must have space for at least count_true elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_select(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    mask_ptr: u64,
    output_ptr: u64,
    prefix_sum_ptr: u64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = kernel_name("masked_select", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&mask_ptr);
        builder.arg(&output_ptr);
        builder.arg(&prefix_sum_ptr);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA masked_select kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

// ============================================================================
// Masked Fill
// ============================================================================

/// Launch masked_fill kernel.
///
/// Fills elements where mask is true with a scalar value.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - input and output must have n elements
pub unsafe fn launch_masked_fill(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    mask_ptr: u64,
    output_ptr: u64,
    fill_value: f64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    let kernel_name = match dtype {
        DType::F32 => "masked_fill_f32",
        DType::F64 => "masked_fill_f64",
        DType::I32 => "masked_fill_i32",
        DType::I64 => "masked_fill_i64",
        #[cfg(feature = "f16")]
        DType::F16 => "masked_fill_f16",
        #[cfg(feature = "f16")]
        DType::BF16 => "masked_fill_bf16",
        #[cfg(feature = "fp8")]
        DType::FP8E4M3 => "masked_fill_fp8_e4m3",
        #[cfg(feature = "fp8")]
        DType::FP8E5M2 => "masked_fill_fp8_e5m2",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "masked_fill",
            });
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&mask_ptr);
        builder.arg(&output_ptr);

        let fill_f32 = fill_value as f32;
        let fill_f64 = fill_value;
        let fill_i32 = fill_value as i32;
        let fill_i64 = fill_value as i64;
        #[cfg(feature = "f16")]
        let fill_f16 = half::f16::from_f64(fill_value).to_bits();
        #[cfg(feature = "f16")]
        let fill_bf16 = half::bf16::from_f64(fill_value).to_bits();
        #[cfg(feature = "fp8")]
        let fill_fp8_e4m3 = crate::dtype::fp8::FP8E4M3::from_f64(fill_value).to_bits();
        #[cfg(feature = "fp8")]
        let fill_fp8_e5m2 = crate::dtype::fp8::FP8E5M2::from_f64(fill_value).to_bits();

        match dtype {
            DType::F32 => builder.arg(&fill_f32),
            DType::F64 => builder.arg(&fill_f64),
            DType::I32 => builder.arg(&fill_i32),
            DType::I64 => builder.arg(&fill_i64),
            #[cfg(feature = "f16")]
            DType::F16 => builder.arg(&fill_f16),
            #[cfg(feature = "f16")]
            DType::BF16 => builder.arg(&fill_bf16),
            #[cfg(feature = "fp8")]
            DType::FP8E4M3 => builder.arg(&fill_fp8_e4m3),
            #[cfg(feature = "fp8")]
            DType::FP8E5M2 => builder.arg(&fill_fp8_e5m2),
            _ => unreachable!(),
        };

        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA masked_fill kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

// ============================================================================
// Broadcast Masked Operations
// ============================================================================

/// Launch broadcast masked_count kernel.
///
/// # Safety
///
/// - mask_ptr must be valid device memory
/// - count_ptr must be valid device memory with 1 u32 element (initialized to 0)
/// - mask_strides_ptr, out_shape_ptr must be valid device memory with ndim u32 elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_count_broadcast(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    mask_ptr: u64,
    count_ptr: u64,
    mask_strides_ptr: u64,
    out_shape_ptr: u64,
    ndim: usize,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, "masked_count_broadcast_kernel")?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&mask_ptr);
        builder.arg(&count_ptr);
        builder.arg(&mask_strides_ptr);
        builder.arg(&out_shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA masked_count_broadcast kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch broadcast masked_prefix_sum kernel.
///
/// # Safety
///
/// - mask_ptr must be valid device memory
/// - prefix_sum_ptr must be valid device memory with n u32 elements
/// - mask_strides_ptr, out_shape_ptr must be valid device memory with ndim u32 elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_prefix_sum_broadcast(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    mask_ptr: u64,
    prefix_sum_ptr: u64,
    mask_strides_ptr: u64,
    out_shape_ptr: u64,
    ndim: usize,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, "masked_prefix_sum_broadcast_kernel")?;

        let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);

        let ndim_u32 = ndim as u32;
        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&mask_ptr);
        builder.arg(&prefix_sum_ptr);
        builder.arg(&mask_strides_ptr);
        builder.arg(&out_shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA masked_prefix_sum_broadcast kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch broadcast masked_select kernel.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - prefix_sum must be precomputed via launch_masked_prefix_sum_broadcast
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_select_broadcast(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    mask_ptr: u64,
    output_ptr: u64,
    prefix_sum_ptr: u64,
    mask_strides_ptr: u64,
    out_shape_ptr: u64,
    ndim: usize,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = format!("masked_select_broadcast_{}", dtype_suffix(dtype)?);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&mask_ptr);
        builder.arg(&output_ptr);
        builder.arg(&prefix_sum_ptr);
        builder.arg(&mask_strides_ptr);
        builder.arg(&out_shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA masked_select_broadcast kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch broadcast masked_fill kernel.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - mask_strides_ptr, out_shape_ptr must be valid device memory with ndim u32 elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_fill_broadcast(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    mask_ptr: u64,
    output_ptr: u64,
    fill_value: f64,
    mask_strides_ptr: u64,
    out_shape_ptr: u64,
    ndim: usize,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    let kernel_name = match dtype {
        DType::F32 => "masked_fill_broadcast_f32",
        DType::F64 => "masked_fill_broadcast_f64",
        DType::I32 => "masked_fill_broadcast_i32",
        DType::I64 => "masked_fill_broadcast_i64",
        #[cfg(feature = "f16")]
        DType::F16 => "masked_fill_broadcast_f16",
        #[cfg(feature = "f16")]
        DType::BF16 => "masked_fill_broadcast_bf16",
        #[cfg(feature = "fp8")]
        DType::FP8E4M3 => "masked_fill_broadcast_fp8_e4m3",
        #[cfg(feature = "fp8")]
        DType::FP8E5M2 => "masked_fill_broadcast_fp8_e5m2",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "masked_fill_broadcast",
            });
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&mask_ptr);
        builder.arg(&output_ptr);

        let fill_f32 = fill_value as f32;
        let fill_f64 = fill_value;
        let fill_i32 = fill_value as i32;
        let fill_i64 = fill_value as i64;
        #[cfg(feature = "f16")]
        let fill_f16 = half::f16::from_f64(fill_value).to_bits();
        #[cfg(feature = "f16")]
        let fill_bf16 = half::bf16::from_f64(fill_value).to_bits();
        #[cfg(feature = "fp8")]
        let fill_fp8_e4m3 = crate::dtype::fp8::FP8E4M3::from_f64(fill_value).to_bits();
        #[cfg(feature = "fp8")]
        let fill_fp8_e5m2 = crate::dtype::fp8::FP8E5M2::from_f64(fill_value).to_bits();

        match dtype {
            DType::F32 => builder.arg(&fill_f32),
            DType::F64 => builder.arg(&fill_f64),
            DType::I32 => builder.arg(&fill_i32),
            DType::I64 => builder.arg(&fill_i64),
            #[cfg(feature = "f16")]
            DType::F16 => builder.arg(&fill_f16),
            #[cfg(feature = "f16")]
            DType::BF16 => builder.arg(&fill_bf16),
            #[cfg(feature = "fp8")]
            DType::FP8E4M3 => builder.arg(&fill_fp8_e4m3),
            #[cfg(feature = "fp8")]
            DType::FP8E5M2 => builder.arg(&fill_fp8_e5m2),
            _ => unreachable!(),
        };

        builder.arg(&mask_strides_ptr);
        builder.arg(&out_shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA masked_fill_broadcast kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Helper to get dtype suffix for kernel name
fn dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F64 => Ok("f64"),
        DType::I32 => Ok("i32"),
        DType::I64 => Ok("i64"),
        #[cfg(feature = "f16")]
        DType::F16 => Ok("f16"),
        #[cfg(feature = "f16")]
        DType::BF16 => Ok("bf16"),
        #[cfg(feature = "fp8")]
        DType::FP8E4M3 => Ok("fp8_e4m3"),
        #[cfg(feature = "fp8")]
        DType::FP8E5M2 => Ok("fp8_e5m2"),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "masked_select_broadcast",
        }),
    }
}
