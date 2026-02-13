//! Shape operation CUDA kernel launchers
//!
//! Provides launchers for shape operations: cat, stack, repeat, pad, roll
//! split and chunk are zero-copy operations using narrow() and don't need kernels.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::CudaRuntime;
use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;

/// Module name for shape operations
pub const SHAPE_MODULE: &str = "shape";

// ============================================================================
// Cat Copy Kernel
// ============================================================================

/// Launch cat_copy kernel to copy one source tensor into the output at the correct offset.
///
/// This kernel copies data from a source tensor to the appropriate position in the
/// concatenated output tensor. It's called once per input tensor in the cat operation.
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - GPU device index
/// * `dtype` - Data type of the tensors
/// * `src_ptr` - Pointer to source tensor data (must be contiguous)
/// * `dst_ptr` - Pointer to output tensor data
/// * `outer_size` - Product of dimensions before cat dimension
/// * `src_cat_size` - Size of source tensor along cat dimension
/// * `dst_cat_size` - Size of output tensor along cat dimension (sum of all inputs)
/// * `cat_offset` - Offset in cat dimension where this source starts
/// * `inner_size` - Product of dimensions after cat dimension
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - src_ptr must point to contiguous data with total elements = outer_size * src_cat_size * inner_size
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_cat_copy(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    src_ptr: u64,
    dst_ptr: u64,
    outer_size: usize,
    src_cat_size: usize,
    dst_cat_size: usize,
    cat_offset: usize,
    inner_size: usize,
) -> Result<()> {
    let total_elements = outer_size * src_cat_size * inner_size;
    if total_elements == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, SHAPE_MODULE)?;
        let func_name = kernel_name("cat_copy", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total_elements);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let outer_u32 = outer_size as u32;
        let src_cat_u32 = src_cat_size as u32;
        let dst_cat_u32 = dst_cat_size as u32;
        let cat_offset_u32 = cat_offset as u32;
        let inner_u32 = inner_size as u32;
        let total_u32 = total_elements as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&dst_ptr);
        builder.arg(&outer_u32);
        builder.arg(&src_cat_u32);
        builder.arg(&dst_cat_u32);
        builder.arg(&cat_offset_u32);
        builder.arg(&inner_u32);
        builder.arg(&total_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA cat_copy kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}

// ============================================================================
// Repeat Kernel
// ============================================================================

/// Launch repeat kernel to tile a tensor along all dimensions.
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - GPU device index
/// * `device` - CUDA device for memory allocation
/// * `dtype` - Data type of the tensors
/// * `src_ptr` - Pointer to source tensor data (must be contiguous)
/// * `dst_ptr` - Pointer to output tensor data
/// * `src_shape` - Shape of source tensor
/// * `out_shape` - Shape of output tensor (src_shape * repeats)
///
/// # Safety
///
/// - All pointers must be valid device memory
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_repeat(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    src_ptr: u64,
    dst_ptr: u64,
    src_shape: &[usize],
    out_shape: &[usize],
) -> Result<()> {
    let total_elements: usize = out_shape.iter().product();
    if total_elements == 0 {
        return Ok(());
    }

    let ndim = src_shape.len();

    // Allocate device memory for shapes
    let shape_bytes = ndim * std::mem::size_of::<u32>();
    let src_shape_ptr = CudaRuntime::allocate(shape_bytes, device)?;
    let out_shape_ptr = CudaRuntime::allocate(shape_bytes, device)?;
    let out_strides_ptr = CudaRuntime::allocate(shape_bytes, device)?;

    // Convert shapes to u32 and copy to device
    let src_shape_u32: Vec<u32> = src_shape.iter().map(|&s| s as u32).collect();
    let out_shape_u32: Vec<u32> = out_shape.iter().map(|&s| s as u32).collect();

    // Compute output strides (row-major)
    let mut out_strides_u32 = vec![0u32; ndim];
    let mut stride = 1u32;
    for i in (0..ndim).rev() {
        out_strides_u32[i] = stride;
        stride *= out_shape_u32[i];
    }

    CudaRuntime::copy_to_device(bytemuck::cast_slice(&src_shape_u32), src_shape_ptr, device)?;
    CudaRuntime::copy_to_device(bytemuck::cast_slice(&out_shape_u32), out_shape_ptr, device)?;
    CudaRuntime::copy_to_device(
        bytemuck::cast_slice(&out_strides_u32),
        out_strides_ptr,
        device,
    )?;

    // Use closure to capture result, ensuring cleanup always runs even if kernel launch fails
    let result: Result<()> = (|| unsafe {
        let module = get_or_load_module(context, device_index, SHAPE_MODULE)?;
        let func_name = kernel_name("repeat", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total_elements);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let total_u32 = total_elements as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&dst_ptr);
        builder.arg(&src_shape_ptr);
        builder.arg(&out_shape_ptr);
        builder.arg(&out_strides_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&total_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA repeat kernel launch failed: {:?}", e)))?;

        Ok(())
    })();

    // Clean up device memory unconditionally (prevents leak on error)
    CudaRuntime::deallocate(src_shape_ptr, shape_bytes, device);
    CudaRuntime::deallocate(out_shape_ptr, shape_bytes, device);
    CudaRuntime::deallocate(out_strides_ptr, shape_bytes, device);

    result
}

// ============================================================================
// Pad Kernel
// ============================================================================

/// Launch pad kernel to add padding around a tensor.
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - GPU device index
/// * `device` - CUDA device for memory allocation
/// * `dtype` - Data type of the tensors
/// * `src_ptr` - Pointer to source tensor data (must be contiguous)
/// * `dst_ptr` - Pointer to output tensor data
/// * `fill_value` - Value to fill in padded regions
/// * `src_shape` - Shape of source tensor
/// * `out_shape` - Shape of output tensor
/// * `pad_before` - Padding before each dimension
///
/// # Safety
///
/// - All pointers must be valid device memory
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_pad(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    src_ptr: u64,
    dst_ptr: u64,
    fill_value: f64,
    src_shape: &[usize],
    out_shape: &[usize],
    pad_before: &[usize],
) -> Result<()> {
    let total_elements: usize = out_shape.iter().product();
    if total_elements == 0 {
        return Ok(());
    }

    let ndim = src_shape.len();

    // Allocate device memory for shapes and padding
    let shape_bytes = ndim * std::mem::size_of::<u32>();
    let src_shape_ptr = CudaRuntime::allocate(shape_bytes, device)?;
    let out_shape_ptr = CudaRuntime::allocate(shape_bytes, device)?;
    let pad_before_ptr = CudaRuntime::allocate(shape_bytes, device)?;

    // Convert to u32 and copy to device
    let src_shape_u32: Vec<u32> = src_shape.iter().map(|&s| s as u32).collect();
    let out_shape_u32: Vec<u32> = out_shape.iter().map(|&s| s as u32).collect();
    let pad_before_u32: Vec<u32> = pad_before.iter().map(|&s| s as u32).collect();

    CudaRuntime::copy_to_device(bytemuck::cast_slice(&src_shape_u32), src_shape_ptr, device)?;
    CudaRuntime::copy_to_device(bytemuck::cast_slice(&out_shape_u32), out_shape_ptr, device)?;
    CudaRuntime::copy_to_device(
        bytemuck::cast_slice(&pad_before_u32),
        pad_before_ptr,
        device,
    )?;

    // Prepare fill values before the closure (all variants needed for borrow lifetime)
    let fill_f32 = fill_value as f32;
    let fill_f64 = fill_value;
    let fill_i32 = fill_value as i32;
    let fill_i64 = fill_value as i64;
    let fill_u32 = fill_value as u32;
    let fill_u64 = fill_value as u64;
    let fill_i16 = fill_value as i16;
    let fill_i8 = fill_value as i8;
    let fill_u16 = fill_value as u16;
    let fill_u8 = fill_value as u8;
    #[cfg(feature = "f16")]
    let fill_f16 = half::f16::from_f64(fill_value);
    #[cfg(feature = "f16")]
    let fill_bf16 = half::bf16::from_f64(fill_value);
    #[cfg(feature = "fp8")]
    let fill_fp8_e4m3 = crate::dtype::FP8E4M3::from_f32(fill_value as f32);
    #[cfg(feature = "fp8")]
    let fill_fp8_e5m2 = crate::dtype::FP8E5M2::from_f32(fill_value as f32);

    // Use closure to capture result, ensuring cleanup always runs even if kernel launch fails
    let result: Result<()> = (|| unsafe {
        let module = get_or_load_module(context, device_index, SHAPE_MODULE)?;
        let func_name = kernel_name("pad", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total_elements);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let total_u32 = total_elements as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&dst_ptr);

        // Pass fill value based on dtype
        match dtype {
            DType::F32 => builder.arg(&fill_f32),
            DType::F64 => builder.arg(&fill_f64),
            DType::I32 => builder.arg(&fill_i32),
            DType::I64 => builder.arg(&fill_i64),
            DType::U32 => builder.arg(&fill_u32),
            DType::U64 => builder.arg(&fill_u64),
            DType::I16 => builder.arg(&fill_i16),
            DType::I8 => builder.arg(&fill_i8),
            DType::U16 => builder.arg(&fill_u16),
            DType::U8 => builder.arg(&fill_u8),
            #[cfg(feature = "f16")]
            DType::F16 => builder.arg(&fill_f16),
            #[cfg(feature = "f16")]
            DType::BF16 => builder.arg(&fill_bf16),
            #[cfg(feature = "fp8")]
            DType::FP8E4M3 => builder.arg(&fill_fp8_e4m3),
            #[cfg(feature = "fp8")]
            DType::FP8E5M2 => builder.arg(&fill_fp8_e5m2),
            _ => {
                return Err(Error::UnsupportedDType { dtype, op: "pad" });
            }
        };

        builder.arg(&src_shape_ptr);
        builder.arg(&out_shape_ptr);
        builder.arg(&pad_before_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&total_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA pad kernel launch failed: {:?}", e)))?;

        Ok(())
    })();

    // Clean up device memory unconditionally (prevents leak on error)
    CudaRuntime::deallocate(src_shape_ptr, shape_bytes, device);
    CudaRuntime::deallocate(out_shape_ptr, shape_bytes, device);
    CudaRuntime::deallocate(pad_before_ptr, shape_bytes, device);

    result
}

// ============================================================================
// Roll Kernel
// ============================================================================

/// Launch roll kernel to shift elements along a dimension with wrapping.
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - GPU device index
/// * `dtype` - Data type of the tensors
/// * `src_ptr` - Pointer to source tensor data (must be contiguous)
/// * `dst_ptr` - Pointer to output tensor data
/// * `outer_size` - Product of dimensions before roll dimension
/// * `dim_size` - Size of roll dimension
/// * `inner_size` - Product of dimensions after roll dimension
/// * `shift` - Amount to shift (positive, normalized to [0, dim_size))
///
/// # Safety
///
/// - All pointers must be valid device memory
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_roll(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    src_ptr: u64,
    dst_ptr: u64,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
    shift: usize,
) -> Result<()> {
    let total_elements = outer_size * dim_size * inner_size;
    if total_elements == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, SHAPE_MODULE)?;
        let func_name = kernel_name("roll", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total_elements);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let outer_u32 = outer_size as u32;
        let dim_u32 = dim_size as u32;
        let inner_u32 = inner_size as u32;
        let shift_u32 = shift as u32;
        let total_u32 = total_elements as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&dst_ptr);
        builder.arg(&outer_u32);
        builder.arg(&dim_u32);
        builder.arg(&inner_u32);
        builder.arg(&shift_u32);
        builder.arg(&total_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA roll kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}
